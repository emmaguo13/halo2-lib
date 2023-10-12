#![allow(non_snake_case)]
#![allow(dead_code)]
use crate::bigint::{CRTInteger, ProperCrtUint};
use crate::fields::fp::Reduced;
use crate::fields::{fp::FpChip, FieldChip};
use halo2_base::utils::BigPrimeField;
use halo2_base::{utils::CurveAffineExt, AssignedValue, Context};

use crate::ecc::{multi_scalar_multiply, EcPoint, EccChip};

/// Computes three vectors of verification scalars \\([u\_{i}^{2}]\\), \\([u\_{i}^{-2}]\\) and \\([s\_{i}]\\) for combined multiscalar multiplication
/// u_{i} is provided as input, assume is checked to be in [0, n - 1]
/// returns (u_{i}^{2}, u_{i}^{-2}, s_{i}) with size k, k, 2^k
pub fn verification_scalars<F: BigPrimeField, SF: BigPrimeField>(
    ctx: &mut Context<F>,
    u: Vec<Reduced<ProperCrtUint<F>, SF>>, // size = k, u_i < n, randomness generated by fiat-shamir transform
    scalar_chip: &FpChip<F, SF>,
) -> (Vec<ProperCrtUint<F>>, Vec<ProperCrtUint<F>>, Vec<ProperCrtUint<F>>) {
    let lg_n = u.len();
    let u_pow_two: Vec<_> = u
        .iter()
        .map(|u_i| {
            scalar_chip.mul(
                ctx,
                CRTInteger::from(ProperCrtUint::from((*u_i).clone())),
                CRTInteger::from(ProperCrtUint::from((*u_i).clone())),
            )
        })
        .collect();

    let sf_one = scalar_chip.load_constant(ctx, SF::ONE);
    let u_invert: Vec<_> = u
        .iter()
        .map(|u_i| scalar_chip.divide(ctx, sf_one.clone(), ProperCrtUint::from((*u_i).clone())))
        .collect();
    let u_inv_pow_two: Vec<_> = u_invert
        .iter()
        .map(|u_i| {
            scalar_chip.mul(ctx, CRTInteger::from((*u_i).clone()), CRTInteger::from((*u_i).clone()))
        })
        .collect();

    // Compute 1/(u_k...u_1)
    let allinv =
        u_invert.iter().fold(sf_one.clone(), |acc, x| scalar_chip.mul(ctx, acc, (*x).clone()));
    // compute s
    let mut s = Vec::with_capacity(2_usize.pow(lg_n as u32));
    println!("s capacity: {}", s.capacity());
    s.push(allinv);
    for i in 1..2_usize.pow(lg_n as u32) {
        let lg_i = (32 - 1 - (i as u32).leading_zeros()) as usize;
        let k = 1 << lg_i;
        let u_lg_i_sq = u_pow_two[(lg_n - 1) - lg_i].clone();
        s.push(scalar_chip.mul(ctx, s[i - k].clone(), u_lg_i_sq));
    }

    (u_pow_two, u_inv_pow_two, s)
}

// CF is the coordinate field of GA
// SF is the scalar field of GA
// p = base field modulus
// n = scalar field modulus
/// follow verification equation in https://doc-internal.dalek.rs/bulletproofs/inner_product_proof/index.html
pub fn inner_product_argument<F: BigPrimeField, CF: BigPrimeField, SF: BigPrimeField, GA>(
    chip: &EccChip<F, FpChip<F, CF>>,
    ctx: &mut Context<F>,
    P: EcPoint<F, <FpChip<F, CF> as FieldChip<F>>::FieldPoint>, // commitment with form P = <a, G> + <b, H> + <a, b>Q
    G: Vec<EcPoint<F, <FpChip<F, CF> as FieldChip<F>>::FieldPoint>>, // size n = 2^k
    H: Vec<EcPoint<F, <FpChip<F, CF> as FieldChip<F>>::FieldPoint>>, // size n = 2^k
    L: Vec<EcPoint<F, <FpChip<F, CF> as FieldChip<F>>::FieldPoint>>, // size k,
    R: Vec<EcPoint<F, <FpChip<F, CF> as FieldChip<F>>::FieldPoint>>, // size k,
    Q: EcPoint<F, <FpChip<F, CF> as FieldChip<F>>::FieldPoint>,
    u: Vec<ProperCrtUint<F>>, // size = k, u_i < n, randomness generated by fiat-shamir transform
    a: ProperCrtUint<F>,      // a < n
    b: ProperCrtUint<F>,      // b < n
    var_window_bits: usize,
) -> AssignedValue<F>
where
    GA: CurveAffineExt<Base = CF, ScalarExt = SF>,
{
    // get FpChip for SF
    let base_chip = chip.field_chip;
    let scalar_chip =
        FpChip::<F, SF>::new(base_chip.range, base_chip.limb_bits, base_chip.num_limbs);

    // validate vector length
    let k = G.len().ilog2();
    assert_eq!(G.len(), H.len());
    assert_eq!(G.len(), 1 << k);
    assert_eq!(u.len() as u32, k);
    assert_eq!(L.len() as u32, k);
    assert_eq!(R.len() as u32, k);

    // validate u, a, b < n
    let a_valid = scalar_chip.enforce_less_than(ctx, a);
    let b_valid = scalar_chip.enforce_less_than(ctx, b);
    // todo: should enforce u_{i} != 0
    let u_valid: Vec<Reduced<ProperCrtUint<F>, SF>> =
        u.iter().map(|u_i| scalar_chip.enforce_less_than(ctx, (*u_i).clone())).collect();

    let (u_pow_two, u_inv_pow_two, s) = verification_scalars(ctx, u_valid, &scalar_chip);

    let a_s: Vec<ProperCrtUint<F>> =
        s.iter().map(|s_i| scalar_chip.mul(ctx, a_valid.0.clone(), (*s_i).clone())).collect();
    let b_invert_s: Vec<ProperCrtUint<F>> =
        s.iter().rev().map(|s_i| scalar_chip.mul(ctx, b_valid.0.clone(), (*s_i).clone())).collect();
    let neg_u_pow_two: Vec<ProperCrtUint<F>> =
        u_pow_two.iter().map(|u_i| scalar_chip.negate(ctx, (*u_i).clone())).collect();
    let neg_u_inv_pow_two: Vec<ProperCrtUint<F>> =
        u_inv_pow_two.iter().map(|u_i| scalar_chip.negate(ctx, (*u_i).clone())).collect();
    let a_b = scalar_chip.mul(ctx, a_valid.0, b_valid.0);
    let p_prime = multi_scalar_multiply::<_, _, GA>(
        base_chip,
        ctx,
        &(G.iter()
            .chain(H.iter())
            .chain(std::iter::once(&Q))
            .chain(L.iter())
            .chain(R.iter())
            .cloned()
            .collect::<Vec<_>>()),
        a_s.iter()
            .chain(b_invert_s.iter())
            .chain(std::iter::once(&a_b))
            .chain(neg_u_pow_two.iter())
            .chain(neg_u_inv_pow_two.iter())
            .map(|x| x.limbs().to_vec())
            .collect(),
        base_chip.limb_bits,
        var_window_bits,
    );

    chip.is_equal(ctx, p_prime, P)
}