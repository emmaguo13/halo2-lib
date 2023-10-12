use crate::{
    gates::{GateInstructions, RangeInstructions, GateChip},
    poseidon::hasher::PoseidonSponge,
    safe_types::{SafeBool, SafeTypeChip},
    utils::{BigPrimeField, fe_to_biguint},
    AssignedValue, Context,
    ScalarField
};

use num_bigint::BigUint;

/// Poseidon circuit transcript
pub struct PoseidonTranscriptChip<F: ScalarField, const T: usize, const RATE: usize> {
    transcript: PoseidonSponge<F, T, RATE>, 
    gate: GateChip<F>,
}

impl<F: ScalarField, const T: usize, const RATE: usize> PoseidonTranscriptChip<F, T, RATE> {
    pub fn new<const R_F: usize, const R_P: usize, const SECURE_MDS: usize>(
        ctx: &mut Context<F>,
    ) -> Self {
        let transcript = PoseidonSponge::<F, T, RATE>::new::<R_F, R_P, 0>(ctx);
        let gate = GateChip::default();

        Self {
            transcript,
            gate
        }
    }

    pub fn absorb(&mut self, items:Vec<AssignedValue<F>>) {
        for item in items {
            self.transcript.update(&[item]);
        }
    }

    pub fn squeeze(&mut self, ctx: &mut Context<F>) -> BigUint{
        let squeezed = self.transcript.squeeze(ctx, &self.gate);
        fe_to_biguint::<F>(squeezed.value())
    }

}


#[cfg(test)]
mod tests;
