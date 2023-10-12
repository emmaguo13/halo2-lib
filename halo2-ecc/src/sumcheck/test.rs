#![allow(non_snake_case)]
use crate::bigint::{self, ProperCrtUint};
use crate::fields::FpStrategy;
use crate::halo2_proofs::{
    arithmetic::CurveAffine,
    halo2curves::bn256::Fr,
    halo2curves::secp256k1::{Fp, Fq, Secp256k1Affine},
};
use crate::secp256k1::{FpChip, FqChip};
use crate::{ecc::EccChip, fields::FieldChip};
use halo2_base::gates::RangeChip;
use halo2_base::halo2_proofs::arithmetic::Field;
use halo2_base::halo2_proofs::halo2curves::group::prime::PrimeCurveAffine;
use halo2_base::poseidon::transcripts::PoseidonTranscriptChip;
use halo2_base::utils::testing::base_test;
use halo2_base::utils::{biguint_to_fe, BigPrimeField};
use halo2_base::{AssignedValue, Context};
use num_bigint::BigUint;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::fs::File;

use super::{verify_sumcheck, UniPoly};

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
    rando: Fq,
) {
    let fq_chip = FqChip::<F>::new(range, params.limb_bits, params.num_limbs);
    let coeffs = input.iter().map(|u| fq_chip.load_private(ctx, *u)).collect::<Vec<_>>();
    let r = fq_chip.load_private(ctx, rando);

    // test initialize
    let unipoly = UniPoly::<F, Fq>::new(coeffs.clone(), &fq_chip);

    let eval_one_res = unipoly.eval_at_one(ctx).value();

    let correct_eval_one_res = input
        .iter()
        .fold(BigUint::from(0 as u32), |acc, x| acc + BigUint::from_bytes_le(&((*x).to_bytes())));

    assert_eq!(eval_one_res, correct_eval_one_res);

    let eval_res = unipoly.evaluate(ctx, r.clone()).value();

    let mut power = fq_chip.load_constant(ctx, Fq::ONE);

    let mut r_powers = Vec::<ProperCrtUint<F>>::new();

    for _ in 0..input.len() {
        r_powers.push(power.clone());
        power = fq_chip.mul(ctx, power.clone(), r.clone());
    }

    let coeff_times_r: Vec<ProperCrtUint<F>> = coeffs
        .iter()
        .enumerate()
        .map(|(i, x)| fq_chip.mul(ctx, r_powers[i].clone(), (*x).clone()))
        .collect();

    let sf_zero = fq_chip.load_constant(ctx, Fq::ZERO);
    let correct_eval_res = coeff_times_r.iter().fold(sf_zero, |acc, x| {
        bigint::ProperCrtUint(fq_chip.add_no_carry(ctx, acc.clone(), x.clone()))
    });

    assert_eq!(eval_res, correct_eval_res.value());
}

pub fn sumcheck_test<F: BigPrimeField>(
    ctx: &mut Context<F>,
    range: &RangeChip<F>,
    params: CircuitParams,
) {
    // todo: potential errors
    // 1. my sumcheck calculations are not correct
    // 2. inputs are not correct - 
    // 3. sumcheck code is not correct

    let claim = <Secp256k1Affine as CurveAffine>::ScalarExt::from(12);
    let mut inputs = Vec::<Vec<ProperCrtUint<F>>>::new();
    let mut rs = Vec::<ProperCrtUint<F>>::new();

    let mut poseidon_transcript = PoseidonTranscriptChip::<F, 3, 2>::new::<8, 57, 0>(ctx);

    let fq_chip = FqChip::<F>::new(range, params.limb_bits, params.num_limbs);

    // round 1 vector
    let v1 = vec![fq_chip.load_private(ctx, Fq::from(3)), fq_chip.load_private(ctx, Fq::from(6))];
    inputs.push(v1.clone());

    // generate first random r by absorbing the coeffs of r1
    poseidon_transcript
        .absorb(v1.iter().map(|v| *v.native()).collect::<Vec<AssignedValue<F>>>());
    let r1 = poseidon_transcript.squeeze(ctx);
    let r1_crt = fq_chip.load_constant_uint(ctx, r1);
    rs.push(r1_crt.clone());

    // round 2 vector
    let r1_times_two = fq_chip.scalar_mul_no_carry(ctx, r1_crt.clone(), 2);

    let r1_times_two_plus_one = fq_chip.add_constant_no_carry(ctx, r1_times_two, Fq::ONE);
    let v2 = vec![
        bigint::ProperCrtUint(r1_times_two_plus_one.clone()),
        bigint::ProperCrtUint(r1_times_two_plus_one),
    ];

    // let v2 = vec![fq_chip.load_private(ctx, biguint_to_fe(&(BigUint::from(2 as u32)*r1.clone() + BigUint::from(1 as u32)))), fq_chip.load_private(ctx, biguint_to_fe(&(BigUint::from(2 as u32)*r1.clone() + BigUint::from(1 as u32))))];
    inputs.push(v2.clone());

    // generate r2
    poseidon_transcript
        .absorb(v2.iter().map(|v| *v.native()).collect::<Vec<AssignedValue<F>>>());
    let r2 = poseidon_transcript.squeeze(ctx);
    let r2_crt = fq_chip.load_constant_uint(ctx, r2);
    rs.push(r2_crt.clone());

    let r1_times_r2 = fq_chip.mul(ctx, r1_crt.clone(), r2_crt.clone());
    let r1_times_r2_plus_r1 = fq_chip.add_no_carry(ctx, r1_times_r2, r1_crt.clone());

    let r2_plus_one = fq_chip.add_constant_no_carry(ctx, r2_crt.clone(), Fq::ONE);

    let v3 = vec![bigint::ProperCrtUint(r1_times_r2_plus_r1), bigint::ProperCrtUint(r2_plus_one)];

    // let v3 = vec![fq_chip.load_private(ctx, biguint_to_fe(&(r2.clone()*r1.clone() + r1.clone()))), fq_chip.load_private(ctx, biguint_to_fe(&(r2.clone() + BigUint::from(1 as u32))))];
    inputs.push(v3.clone());

    poseidon_transcript
        .absorb(v3.iter().map(|v| *((*v).native())).collect::<Vec<AssignedValue<F>>>());
    let r3 = poseidon_transcript.squeeze(ctx);
    let r3_crt = fq_chip.load_constant_uint(ctx, r3);
    rs.push(r3_crt);

    // let eval_v2_ml = fq_chip.mul(ctx, r2_crt.clone(), v2[1].clone());
    // let eval_v2 = fq_chip.add_no_carry(ctx, eval_v2_ml, v2[0].clone());

    // let v3_one_eval = fq_chip.add_no_carry(ctx, v3[0].clone(), v3[1].clone());
    // let v3_one_zero = fq_chip.add_no_carry(ctx, v3_one_eval, v3[0].clone());

    // fq_chip.assert_equal(ctx, bigint::ProperCrtUint(v3_one_zero), bigint::ProperCrtUint(eval_v2));

    let claim = fq_chip.load_private(ctx, claim);

    let polys = inputs
        .iter()
        .map(|coeffs| UniPoly::new((*coeffs).clone(), &fq_chip))
        .collect::<Vec<UniPoly<F, Fq>>>();

    let (a, b) = verify_sumcheck::<F, Fq, 3, 2>(
        ctx,
        claim,
        polys.len(),
        polys,
        rs,
        &fq_chip,
        &mut poseidon_transcript,
    );
}

fn random_coeffs(len: usize, rng: &mut StdRng) -> Vec<Fq> {
    (0..len)
        .map(|_| <Secp256k1Affine as CurveAffine>::ScalarExt::random(rng.clone()))
        .collect::<Vec<_>>()
}

#[test]
fn secp256k1_poly_test() {
    let mut rng = StdRng::seed_from_u64(0);

    let rando = <Secp256k1Affine as CurveAffine>::ScalarExt::random(rng.clone());

    let input = random_coeffs(3, &mut rng);

    let path = "configs/secp256k1/ecdsa_circuit.config";
    let params: CircuitParams = serde_json::from_reader(
        File::open(path).unwrap_or_else(|e| panic!("{path} does not exist: {e:?}")),
    )
    .unwrap();

    let res = base_test()
        .k(params.degree)
        .lookup_bits(params.lookup_bits)
        .run(|ctx, range| unipoly_test(ctx, range, params, input, rando));
    // assert_eq!(res, Fr::ONE);
}

#[test]
fn secp256k1_sumcheck_test() {
    // simple sumcheck test with 3 dimensional input

    let path = "configs/secp256k1/ecdsa_circuit.config";
    let params: CircuitParams = serde_json::from_reader(
        File::open(path).unwrap_or_else(|e| panic!("{path} does not exist: {e:?}")),
    )
    .unwrap();

    let res = base_test()
        .k(params.degree)
        .lookup_bits(params.lookup_bits)
        .run(|ctx, range| sumcheck_test(ctx, range, params));
    // assert_eq!(res, Fr::ZERO);
}
