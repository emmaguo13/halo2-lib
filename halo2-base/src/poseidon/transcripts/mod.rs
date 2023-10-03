use crate::{
    gates::{GateInstructions, RangeInstructions},
    poseidon::hasher::PoseidonSponge,
    safe_types::{SafeBool, SafeTypeChip},
    utils::BigPrimeField,
    AssignedValue, Context,
    QuantumCell::Constant,
    ScalarField,
};

// TODO: set everything up with impls, write all function defs
// TODO: emulate the NOVA transcript for scalars

const POSEIDON_STATE_SIZE: usize = 64;

/// Poseidon circuit transcript
pub struct PoseidonTranscriptChip<F: ScalarField, const T: usize, const RATE: usize> {
    // TODO: check
    //   round: u16,
    //   state: [u8; POSEIDON_STATE_SIZE], // TODO: change state size, maybe don't need because of PoseidonState?
    transcript: PoseidonSponge<F, T, RATE>, // TODO: maybe include the <> stuff?
                                            //   _p: PhantomData<G>, // TODO: what is this?
}

// Is this supposed to be a prime field ?
// pub struct Challenge<F: PrimeField, N: PrimeField> {
//     le_bits: Vec<Witness<N>>,
//     scalar: AssignedBase<F, N>,
// }

impl<F: ScalarField, const T: usize, const RATE: usize> PoseidonTranscriptChip<F, T, RATE> {
    pub fn new<const R_F: usize, const R_P: usize, const SECURE_MDS: usize>(
        ctx: &mut Context<F>,
    ) -> Self {
        let mut poseidon_instance = PoseidonSponge::<F, T, RATE>::new::<R_F, R_P, 0>(ctx);
        // let input = [PERSONA_TAG, label].concat();
        // let output = compute_updated_state(keccak_instance.clone(), &input);

        Self {
            //   round: 0u16,
            //   state: output,
            transcript: poseidon_instance,
            //   _p: PhantomData,
        }
    }

    // TODO: fix the inputs?
    // pub fn absorb(&mut self, ctx: &mut Context<F>, label:Vec<F>, o:Vec<F>) {
    //     self.transcript.update(&ctx.assign_witnesses(label));
    //     self.transcript.update(&ctx.assign_witnesses(label));
    // }

    // TODO: either use the sponge squeeze, or write the implementation -> try!
}

pub trait PoseidonTranscriptInstructions<F: ScalarField> {
    fn absorb_field_element(
        &mut self,
        ctx: &mut Context<F>,
        elements: &[AssignedValue<F>],
    )
    where
        F: BigPrimeField;
}

impl<F: ScalarField, const T: usize, const RATE: usize> PoseidonTranscriptInstructions<F>
    for PoseidonTranscriptChip<F, T, RATE>
{
    fn absorb_field_element(
        &mut self,
        ctx: &mut Context<F>,
        elements: &[AssignedValue<F>], // vector of scalar field elements
    ) 
    where
        F: BigPrimeField, 
    {
        self.transcript.update(elements);
    }

    // pub fn squeeze(
    //     &mut self,
    //     ctx: &mut Context<F>,
    //     gate: &impl GateInstructions<F>,
    // ) -> AssignedValue<F> {

    // fn squeeze_field_element(
    //     &self,
    //     ctx: &mut Context<F>,
    // ) -> AssignedValue<F>
    // where
    //     F: BigPrimeField, 
    // {
    //     self.transcript.update(elements);
    // }

}

// fn absorb<T: TranscriptReprTrait<G>>(&mut self, label: &'static [u8], o: &T) {
//     self.transcript.update(label);
//     self.transcript.update(&o.to_transcript_bytes());
//   }

#[cfg(test)]
mod tests;
