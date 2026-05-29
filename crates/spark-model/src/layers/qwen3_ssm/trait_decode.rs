// SPDX-License-Identifier: AGPL-3.0-only

//! TransformerLayer::decode (single-token).

use super::*;

impl Qwen3SsmLayer {
    pub(super) fn decode_inner(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        _seq_len: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let debug = tracing::enabled!(tracing::Level::DEBUG);
        let trace = false;

        // ATLAS_SSM_SPLIT: complete two-sync mixer-vs-MoE split for one
        // SSM layer-seq call (unlike the partial per-op prof! macro). One
        // sync around the whole ssm_forward (mixer) span, one around the
        // post-attn-norm + MoE span; emits a single SSMSPLIT line so the
        // mixer:moe ratio isn't biased by per-op sync overhead.
        let ssm_split =
            !ctx.profile && std::env::var("ATLAS_SSM_SPLIT").is_ok_and(|v| v == "1" || v == "true");

        let ssm_state = state
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState"))?;

        let normed = ctx.buffers.norm_output();
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            hidden,
            &self.input_norm,
            normed,
            residual,
            1,
            h as u32,
            eps,
            stream,
        )?;
        if debug {
            ctx.gpu.synchronize(stream)?;
            Self::debug_bf16(ctx.gpu, "pre-norm", normed, 4);
        }

        let ssm_split_t0 = if ssm_split {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };
        let ssm_out = self.ssm_forward(normed, ssm_state, ctx, stream, trace)?;
        let ssm_split_mixer_us = match ssm_split_t0 {
            Some(t0) => {
                ctx.gpu.synchronize(stream)?;
                Some(t0.elapsed().as_micros())
            }
            None => None,
        };
        if debug {
            ctx.gpu.synchronize(stream)?;
            Self::debug_bf16(ctx.gpu, "ssm-out", ssm_out, 4);
        }

        // Profile: time SSM vs MoE separately
        if ctx.profile {
            use std::time::Instant;
            ctx.gpu.synchronize(stream)?;
            let t0 = Instant::now();

            let normed2 = ctx.buffers.norm_output();
            ops::residual_add_rms_norm(
                ctx.gpu,
                self.residual_add_rms_norm_k,
                hidden,
                ssm_out,
                &self.post_attn_norm,
                normed2,
                residual,
                1,
                h as u32,
                eps,
                stream,
            )?;
            let moe_out = self.ffn.forward(normed2, ctx, stream)?;
            ctx.gpu.synchronize(stream)?;
            let moe_us = t0.elapsed().as_micros();
            tracing::info!("  SSM-MoE: {:.1}ms", moe_us as f64 / 1000.0);

            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden,
                moe_out,
                h as u32,
                stream,
            )?;
            return Ok(());
        }

        let normed2 = ctx.buffers.norm_output();
        let ssm_split_moe_t0 = ssm_split_mixer_us.map(|_| std::time::Instant::now());
        ops::residual_add_rms_norm(
            ctx.gpu,
            self.residual_add_rms_norm_k,
            hidden,
            ssm_out,
            &self.post_attn_norm,
            normed2,
            residual,
            1,
            h as u32,
            eps,
            stream,
        )?;
        if debug {
            ctx.gpu.synchronize(stream)?;
            Self::debug_bf16(ctx.gpu, "post-ssm-residual", residual, 4);
            Self::debug_bf16(ctx.gpu, "post-ssm-hidden", hidden, 4);
            Self::debug_bf16(ctx.gpu, "moe-input-normed", normed2, 4);
        }

        let moe_out = self.ffn.forward(normed2, ctx, stream)?;
        if let (Some(mix), Some(t0)) = (ssm_split_mixer_us, ssm_split_moe_t0) {
            ctx.gpu.synchronize(stream)?;
            tracing::info!(
                "SSMSPLIT mixer={}us moe={}us",
                mix,
                t0.elapsed().as_micros()
            );
        }
        if debug {
            ctx.gpu.synchronize(stream)?;
            Self::debug_bf16(ctx.gpu, "moe-output", moe_out, 8);
        }
        ops::residual_add(
            ctx.gpu,
            self.residual_add_k,
            hidden,
            moe_out,
            h as u32,
            stream,
        )?;
        if debug {
            ctx.gpu.synchronize(stream)?;
            Self::debug_bf16(ctx.gpu, "final-hidden", hidden, 4);
        }

        Ok(())
    }
}
