use halo2_base::{
    gates::flex_gate::{GateChip, GateInstructions},
    utils::{CurveAffineExt, ScalarField, BigPrimeField},
    halo2_proofs,
    AssignedValue,
    Context,
    QuantumCell::{Constant, Existing, Witness, WitnessFraction},
};

use crate::halo2_proofs::{
    arithmetic::CurveAffine,
    halo2curves::bn256::Fr,
    halo2curves::secp256k1::{Fp, Fq, Secp256k1Affine},
};

use crate::bigint::{CRTInteger, ProperCrtUint};
use crate::fields::{fp::FpChip, FieldChip};
use crate::ecc::{multi_scalar_multiply, EcPoint, EccChip};
use serde::{Deserialize, Serialize};
use std::fs::File;

// TODO: figure out the field stuff, ask sen the question!

pub struct UniPoly<F: BigPrimeField, SF: BigPrimeField> {
    coeffs: Vec<ProperCrtUint<F>>,
    scalar_chip: &FpChip<F, SF>
}

// TODO: new function and make sure the field things are correct
impl<F: BigPrimeField, SF: BigPrimeField> UniPoly<F, SF> {
    pub fn new(
        // ctx: &mut Context<F>,
        &self,
        coeffs: Vec<ProperCrtUint<F>>,
        scalar_chip:&FpChip<F, SF>
    ) {
        self.coeffs = coeffs; 
        self.scalar_chip = scalar_chip;
    }
    pub fn eval_at_one(&self, ctx: &mut Context<F>) -> AssignedValue<F> {
        let sf_zero = scalar_chip.load_constant(ctx, SF::ZERO);
        let eval_sum = self.coeffs.iter().fold(sf_zero.clone(), |acc, x| scalar_chip.add_no_carry(ctx, acc, (*x).clone()));
        eval_sum
    }

    pub fn evaluate(&self, ctx: &mut Context<F>, r: ProperCrtUint<F>) -> AssignedValue<F> {
        let mut eval = self.coeffs[0];
        let mut power = r;
        for coeff in self.coeffs.iter().skip(1) {
            let inter_mul = self.scalar_chip.mul(ctx, coeff, power);
            eval = self.scalar_chip.add_no_carry(ctx, eval, inter_mul);
            power = self.scalar_chip.mul_no_carry(ctx, power, r);
        }
        eval
    }
}


// take in compressed polynomials -- how are multilinear polynomials represented in halo2?
// TODO: not sure if i need SF or GA
// fn verify_sumcheck<F: BigPrimeField, CF: BigPrimeField, SF: BigPrimeField, GA>(
//     ctx: &mut Context<F>,
//     claim: ProperCrtUint<F>,
//     num_rounds: usize,
//     // degree_bound: usize,
//     polys: Vec<UniPoly<F, SF>>, // TODO: do i need the poly to contain CrtUnits?
//     scalar_chip: &FpChip<F, SF>, // TODO: understand
//     // transcript: &mut T // impl TranscriptInstruction<F, TccChip = Self>,
// ) -> (ProperCrtUint<F>, Vec<ProperCrtUint<F>>)
// // where T: TranscriptRead<GA,L>, GA: CurveAffineExt<Base = CF, ScalarExt = SF>,
// {
//     // verify that there is a univariate polynomial for each round
//     assert_eq!(num_rounds, polys.len());

//     let mut e = claim;

//     for i in 0..num_rounds {
//         let poly = polys[i];

//         // TODO: verify degree bound
//         let zero_one_sum = scalar_chip.add(ctx, poly.coeffs[0], poly.eval_at_one(ctx));
//         let zero_one_eq = scalar_chip.is_equal(ctx, zero_one_sum, claim);

//         // TODO : make this compatible with transcript
//         // transcript.absorb(&poly);

//         //derive the verifier's challenge for the next round
//         // let r_i = transcript.squeeze();
//         // TODO: we have to make sure its a ProperCrtUint<F>
  
//         // evaluate the claimed degree-ell polynomial at r_i
//         // e = poly.evaluate(ctx, &r_i);

//     }

//     // TODO: make sure we're doing the right thing at the end.

// }

#[cfg(test)]
mod test;
// mod tests {
//     use super::*;

//     #[derive(Clone, Copy, Debug, Serialize, Deserialize)]
//     pub struct CircuitParams {
//         strategy: FpStrategy,
//         degree: u32,
//         num_advice: usize,
//         num_lookup_advice: usize,
//         num_fixed: usize,
//         lookup_bits: usize,
//         limb_bits: usize,
//         num_limbs: usize,
//     }

//     pub fn unipoly_test<F: BigPrimeField>(
//         ctx: &mut Context<F>,
//         range: &RangeChip<F>,
//         params: CircuitParams,
//         input: Vec<Fq>,
//     ) -> F {
//         let fq_chip = FqChip::<F>::new(range, params.limb_bits, params.num_limbs);
//         let coeffs = input.iter().map(|u| fq_chip.load_private(ctx, *u)).collect::<Vec<_>>();

//         let fp_chip = FpChip::<F>::new(range, params.limb_bits, params.num_limbs);

//         // test initialize
//         let unipoly = UniPoly::<F>::new::<F>(coeffs, &fp_chip);
//     }

//     pub fn run_test(input: Vec<Fq>) {
//         let path = "configs/secp256k1/ecdsa_circuit.config";
//         let params: CircuitParams = serde_json::from_reader(
//             File::open(path).unwrap_or_else(|e| panic!("{path} does not exist: {e:?}")),
//         )
//         .unwrap();
    
//         let res = base_test()
//             .k(params.degree)
//             .lookup_bits(params.lookup_bits)
//             .run(|ctx, range| unipoly_test(ctx, range, params, input));
//         assert_eq!(res, Fr::ONE);
//     }


//     fn random_coeffs(len: usize, rng: &mut StdRng) -> Vec<Fq> {
//         (0..len)
//         .map(|_| <Secp256k1Affine as CurveAffine>::ScalarExt::random(rng.clone()))
//         .collect::<Vec<_>>()
//     }

//     #[test]
//     fn secp256k1_poly_test() {
//         let mut rng = StdRng::seed_from_u64(0);
//         let input = random_coeffs(10, &mut rng);
//         run_test(input);
//     }
// }
