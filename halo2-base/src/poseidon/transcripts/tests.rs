use super::*;

use crate::{
    gates::{flex_gate::threads::SinglePhaseCoreManager, GateChip},
    halo2_proofs::halo2curves::bn256::Fr,
    utils::ScalarField,
    AssignedValue, Context,
};

fn initialize_transcript<
    F: ScalarField,
    const T: usize,
    const RATE: usize,
    const R_F: usize,
    const R_P: usize,
>() {
    let mut pool = SinglePhaseCoreManager::new(true, Default::default());

    let ctx = pool.main();

    let mut poseidon_transcript = PoseidonTranscriptChip::<F, T, RATE>::new::<R_F, R_P, 0>(ctx);
}

#[test]
fn init_transcript() {
    initialize_transcript::<Fr, 3, 2, 8, 57>();
}
