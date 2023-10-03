#![allow(non_snake_case)]
use crate::fields::FpStrategy;
use crate::halo2_proofs::{
    arithmetic::CurveAffine,
    halo2curves::bn256::Fr,
    halo2curves::secp256k1::{Fp, Fq, Secp256k1Affine},
};
use crate::ipa::inner_product_argument::inner_product_argument;
use crate::secp256k1::{FpChip, FqChip};
use crate::{ecc::EccChip, fields::FieldChip};
use halo2_base::gates::RangeChip;
use halo2_base::halo2_proofs::arithmetic::Field;
use halo2_base::halo2_proofs::halo2curves::group::prime::PrimeCurveAffine;
use halo2_base::utils::testing::base_test;
use halo2_base::utils::BigPrimeField;
use halo2_base::Context;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::fs::File;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct CircuitParams {
    strategy: FpStrategy,
    degree: u32,
    num_advice: usize,
    num_lookup_advice: usize,
    num_fixed: usize,
    lookup_bits: usize,
    limb_bits: usize,
    num_limbs: usize,
}

pub fn unipoly_test<F: BigPrimeField>(
    ctx: &mut Context<F>,
    range: &RangeChip<F>,
    params: CircuitParams,
    input: Vec<Fq>,
) -> F {
    let fq_chip = FqChip::<F>::new(range, params.limb_bits, params.num_limbs);
    let coeffs = input.iter().map(|u| fq_chip.load_private(ctx, *u)).collect::<Vec<_>>();

    let fp_chip = FpChip::<F>::new(range, params.limb_bits, params.num_limbs);

    // test initialize
    let unipoly = UniPoly::<F, >::new::<F, Fq>(coeffs, &fp_chip);
}

pub fn run_test(input: Vec<Fq>) {
    let path = "configs/secp256k1/ecdsa_circuit.config";
    let params: CircuitParams = serde_json::from_reader(
        File::open(path).unwrap_or_else(|e| panic!("{path} does not exist: {e:?}")),
    )
    .unwrap();

    let res = base_test()
        .k(params.degree)
        .lookup_bits(params.lookup_bits)
        .run(|ctx, range| unipoly_test(ctx, range, params, input));
    assert_eq!(res, Fr::ONE);
}


fn random_coeffs(len: usize, rng: &mut StdRng) -> Vec<Fq> {
    (0..len)
    .map(|_| <Secp256k1Affine as CurveAffine>::ScalarExt::random(rng.clone()))
    .collect::<Vec<_>>()
}

#[test]
fn secp256k1_poly_test() {
    let mut rng = StdRng::seed_from_u64(0);
    let input = random_coeffs(10, &mut rng);
    run_test(input);
}