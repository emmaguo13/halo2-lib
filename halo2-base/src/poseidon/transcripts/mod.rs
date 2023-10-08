use crate::{
    gates::{GateInstructions, RangeInstructions, GateChip},
    poseidon::hasher::PoseidonSponge,
    safe_types::{SafeBool, SafeTypeChip},
    utils::{BigPrimeField, fe_to_biguint},
    AssignedValue, Context,
    ScalarField
};

use num_bigint::BigUint;

const POSEIDON_STATE_SIZE: usize = 64;

// TODO: write tests, figure out the field stuff, figure out interaction with sumcheck

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

    // TODO: either use the sponge squeeze, or write the implementation -> try!
}

// pub trait PoseidonTranscriptInstructions<F: ScalarField> {
//     fn absorb_field_element(
//         &mut self,
//         ctx: &mut Context<F>,
//         elements: &[AssignedValue<F>],
//     )
//     where
//         F: BigPrimeField;
// }

// impl<F: ScalarField, const T: usize, const RATE: usize> PoseidonTranscriptInstructions<F>
//     for PoseidonTranscriptChip<F, T, RATE>
// {
//     fn absorb_field_element(
//         &mut self,
//         ctx: &mut Context<F>,
//         elements: &[AssignedValue<F>], // vector of scalar field elements
//     ) 
//     where
//         F: BigPrimeField, 
//     {
//         self.transcript.update(elements);
//     }

//     // pub fn squeeze(
//     //     &mut self,
//     //     ctx: &mut Context<F>,
//     //     gate: &impl GateInstructions<F>,
//     // ) -> AssignedValue<F> {

//     // fn squeeze_field_element(
//     //     &self,
//     //     ctx: &mut Context<F>,
//     // ) -> AssignedValue<F>
//     // where
//     //     F: BigPrimeField, 
//     // {
//     //     self.transcript.update(elements);
//     // }

// }

// fn absorb<T: TranscriptReprTrait<G>>(&mut self, label: &'static [u8], o: &T) {
//     self.transcript.update(label);
//     self.transcript.update(&o.to_transcript_bytes());
//   }

#[cfg(test)]
mod tests;
