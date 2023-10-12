use super::*;

use crate::{
    gates::{flex_gate::threads::SinglePhaseCoreManager, GateChip},
    halo2_proofs::halo2curves::bn256::Fr,
    utils::ScalarField,
    AssignedValue, Context,
};

use rand::Rng;

fn transcript_verification<
    F: ScalarField,
    const T: usize,
    const RATE: usize,
    const R_F: usize,
    const R_P: usize,
>() {
    let mut pool = SinglePhaseCoreManager::new(true, Default::default());

    let ctx = pool.main();

    let mut poseidon_transcript = PoseidonTranscriptChip::<F, T, RATE>::new::<R_F, R_P, 0>(ctx);
    let mut poseidon_transcript2 = PoseidonTranscriptChip::<F, T, RATE>::new::<R_F, R_P, 0>(ctx);

    let absorptions: Vec<Vec<F>> = random_nested_list_f(10, 5);

    for absorption in absorptions.clone() {
        poseidon_transcript.absorb(ctx.assign_witnesses(absorption.clone()));
        poseidon_transcript2.absorb(ctx.assign_witnesses(absorption.clone()));

        assert_eq!(poseidon_transcript.squeeze(ctx), poseidon_transcript2.squeeze(ctx));
    }

    for absorption in absorptions.clone() {
        poseidon_transcript.absorb(ctx.assign_witnesses(absorption.clone()));
        poseidon_transcript2.absorb(ctx.assign_witnesses(absorption.clone()));
    }

    assert_eq!(poseidon_transcript.squeeze(ctx), poseidon_transcript2.squeeze(ctx));
}

fn random_nested_list_f<F: ScalarField>(len: usize, max_sub_len: usize) -> Vec<Vec<F>> {
    let mut rng = rand::thread_rng();
    let mut list = Vec::new();
    for _ in 0..len {
        let len = rng.gen_range(0..=max_sub_len);
        let mut sublist = Vec::new();

        for _ in 0..len {
            sublist.push(F::random(&mut rng));
        }
        list.push(sublist);
    }
    list
}

#[test]
fn test_transcript() {
    transcript_verification::<Fr, 3, 2, 8, 57>();
}
