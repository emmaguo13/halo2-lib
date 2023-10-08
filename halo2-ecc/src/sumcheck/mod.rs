use halo2_base::{
    gates::flex_gate::{GateChip, GateInstructions},
    halo2_proofs,
    poseidon::transcripts::PoseidonTranscriptChip,
    utils::{fe_to_biguint, BigPrimeField, CurveAffineExt, ScalarField},
    AssignedValue, Context,
    QuantumCell::{Constant, Existing, Witness, WitnessFraction},
};
use rand_core::le;

use crate::{
    bigint,
    halo2_proofs::{
        arithmetic::CurveAffine,
        halo2curves::bn256::Fr,
        halo2curves::secp256k1::{Fp, Fq, Secp256k1Affine},
    },
};

use crate::bigint::{CRTInteger, ProperCrtUint};
use crate::ecc::{multi_scalar_multiply, EcPoint, EccChip};
use crate::fields::{fp::FpChip, FieldChip};
use serde::{Deserialize, Serialize};
use std::fs::File;

#[derive(Clone)]
pub struct UniPoly<'a, F: BigPrimeField, SF: BigPrimeField> {
    coeffs: Vec<ProperCrtUint<F>>,
    scalar_chip: &'a FpChip<'a, F, SF>,
}

impl<'a, F: BigPrimeField, SF: BigPrimeField> UniPoly<'a, F, SF> {
    pub fn new(
        // ctx: &mut Context<F>,
        coeffs: Vec<ProperCrtUint<F>>,
        scalar_chip: &'a FpChip<'a, F, SF>,
    ) -> Self {
        Self { coeffs, scalar_chip }
    }
    pub fn eval_at_one(&self, ctx: &mut Context<F>) -> ProperCrtUint<F> {
        let sf_zero = self.scalar_chip.load_constant(ctx, SF::ZERO);
        let eval_sum = self.coeffs.iter().fold(sf_zero.clone(), |acc, x| {
            bigint::ProperCrtUint(self.scalar_chip.add_no_carry(ctx, acc, (*x).clone()))
        });
        eval_sum
    }

    pub fn evaluate(&self, ctx: &mut Context<F>, r: ProperCrtUint<F>) -> ProperCrtUint<F> {
        let mut power = r.clone();
        let mut intermediates: Vec<ProperCrtUint<F>> = Vec::<ProperCrtUint<F>>::new();

        let sf_zero = self.scalar_chip.load_constant(ctx, SF::ZERO);

        for coeff in self.coeffs.iter() {
            let inter_mul = self.scalar_chip.mul(ctx, coeff, power.clone());
            intermediates.push(inter_mul);

            let temp_power = self.scalar_chip.mul(ctx, power.clone(), r.clone());
            power = temp_power;
        }

        intermediates.iter().fold(sf_zero.clone(), |acc, x| {
            bigint::ProperCrtUint(self.scalar_chip.add_no_carry(ctx, acc, (*x).clone()))
        })
    }
}

// TODO: not sure if i need SF or GA
// we need to impl transcript with the scalar field of the multilinear polys
fn verify_sumcheck<F: BigPrimeField, SF: BigPrimeField, const T: usize, const RATE: usize>(
    ctx: &mut Context<F>,
    claim: ProperCrtUint<F>,
    num_rounds: usize,
    // degree_bound: usize,
    polys: Vec<UniPoly<F, SF>>,
    scalar_chip: &FpChip<F, SF>,
    transcript: &mut PoseidonTranscriptChip<F, T, RATE>, // impl PoseidonInstruction<F, TccChip = Self>, // todo: impl instructions
) -> (ProperCrtUint<F>, Vec<ProperCrtUint<F>>) {
    // verify that there is a univariate polynomial for each round
    assert_eq!(num_rounds, polys.len());

    let mut e = claim.clone();
    let mut r: Vec<ProperCrtUint<F>> = Vec::new();

    for i in 0..num_rounds {
        let poly = polys[i].clone();
        let eval_one = poly.eval_at_one(ctx);

        // TODO: verify degree bound
        let zero_one_sum = scalar_chip.add_no_carry(ctx, poly.coeffs[0].clone(), eval_one);
        let valid_sum = scalar_chip.enforce_less_than(ctx, bigint::ProperCrtUint(zero_one_sum));

        let zero_one_eq = scalar_chip.is_equal(ctx, valid_sum, claim.clone());

        let native_coeffs = poly.coeffs.iter().map(|u| *u.native()).collect::<Vec<_>>();
        transcript.absorb(native_coeffs);

        //derive the verifier's challenge for the next round
        let r_i = transcript.squeeze(ctx);

        let r_i_crt = scalar_chip.load_constant_uint(ctx, r_i.clone());

        r.push(r_i_crt.clone());

        // evaluate the claimed degree-ell polynomial at r_i
        e = poly.evaluate(ctx, r_i_crt.clone());
    }

    (e, r)
}

#[cfg(test)]
mod test;
