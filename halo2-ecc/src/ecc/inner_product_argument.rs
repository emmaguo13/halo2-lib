use crate::bigint::{CRTInteger, ProperCrtUint};
use crate::fields::fp::Reduced;
use crate::fields::{fp::FpChip, FieldChip, PrimeField};
use halo2_base::{utils::CurveAffineExt, AssignedValue, Context};

use super::{multi_scalar_multiply, EcPoint, EccChip};

fn inner_product<'range, F: PrimeField, SF: PrimeField>(ctx: &mut Context<F>, scalar_chip: &FpChip<'range, F, SF>, a: Vec<Reduced<ProperCrtUint<F>, SF>>, b: Vec<Reduced<ProperCrtUint<F>, SF>>) -> Vec<Reduced<ProperCrtUint<F>, SF>> {
    assert_eq!(a.len(), b.len());

    let a_b = a
        .iter()
        .zip(b.iter())
        .map(|(a_i, b_i)| {
            scalar_chip.mul(
                ctx,
                CRTInteger::from(ProperCrtUint::from((*a_i).clone())),
                CRTInteger::from(ProperCrtUint::from((*b_i).clone())),
            )
        }).fold(CRTInteger::zero(), |acc, x| scalar_chip.add_no_carry(ctx, CRTInteger::from(ProperCrtUint::from((*acc).clone())), CRTInteger::from(ProperCrtUint::from((*x).clone()))));
    
    a_b
}

pub fn create_proof<F: PrimeField, CF: PrimeField, SF: PrimeField, GA>(
    chip: &EccChip<F, FpChip<F, CF>>,
    ctx: &mut Context<F>,
    Q: EcPoint<F, <FpChip<F, CF> as FieldChip<F>>::FieldPoint>, // EC point
    mut G: Vec<EcPoint<F, <FpChip<F, CF> as FieldChip<F>>::FieldPoint>>,
    mut H: Vec<EcPoint<F, <FpChip<F, CF> as FieldChip<F>>::FieldPoint>>,
    mut a_vec: Vec<ProperCrtUint<F>>,
    mut b_vec: Vec<ProperCrtUint<F>>,
    u: Vec<ProperCrtUint<F>>, // todo: get rid of this
    // transcript: &mut Transcript, // todo: implement our own ver of merlin transcript
    var_window_bits: usize,
) where
    GA: CurveAffineExt<Base = CF, ScalarExt = SF>,
{
    // get FpChip for SF
    let base_chip = chip.field_chip;
    let scalar_chip =
        FpChip::<F, SF>::new(base_chip.range, base_chip.limb_bits, base_chip.num_limbs);

    // validate vector length
    let n = G.len();
    let k = G.len().ilog2();
    assert_eq!(n, H.len());
    assert_eq!(n, a_vec.len());
    assert_eq!(n, b_vec.len());
    assert_eq!(n, 1 << k); // does the power of 2 check

    // validate u, a, b < n
    let u_valid: Vec<Reduced<ProperCrtUint<F>, SF>> =
        u.iter().map(|u_i| scalar_chip.enforce_less_than(ctx, (*u_i).clone())).collect();
    let a_valid: Vec<Reduced<ProperCrtUint<F>, SF>> =
        a_vec.iter().map(|a_i| scalar_chip.enforce_less_than(ctx, (*a_i).clone())).collect();
    let b_valid: Vec<Reduced<ProperCrtUint<F>, SF>> =
        b_vec.iter().map(|b_i| scalar_chip.enforce_less_than(ctx, (*b_i).clone())).collect();

    // Init L and R vectors to store the halves
    let lg_n = n.next_power_of_two().trailing_zeros() as usize; // 4 is the next power of two of 3
    let mut L_vec = Vec::with_capacity(lg_n);
    let mut R_vec = Vec::with_capacity(lg_n);

    // If it's the first iteration, unroll the Hprime = H*y_inv scalar mults
    // into multiscalar muls, for performance.
    // todo: actually implement this part ^
    if n != 1 {
        n = n / 2;
        let (a_L, a_R) = a_valid.split_at_mut(n);
        let (b_L, b_R) = b_valid.split_at_mut(n);
        let (G_L, G_R) = G.split_at_mut(n);
        let (H_L, H_R) = H.split_at_mut(n);

        // todo: fix
        let c_L = inner_product(ctx, &scalar_chip, a_L, b_R);
        let c_R = inner_product(ctx, &scalar_chip, a_R, b_L);

        let L = multi_scalar_multiply::<_, _, GA>(
            base_chip,
            ctx,
            &(G_R.iter().chain(H_L.iter()).chain(std::iter::once(&Q)).cloned().collect::<Vec<_>>()), // P
            a_L.iter()
                .chain(b_R.iter())
                .chain(std::iter::once(&c_L))
                .map(|x| x.limbs().to_vec())
                .collect(), // scalar
            base_chip.limb_bits, // max bits
            var_window_bits,     // window bits
        );

        let R = multi_scalar_multiply::<_, _, GA>(
            base_chip,
            ctx,
            &(G_L.iter().chain(H_R.iter()).chain(std::iter::once(&Q)).cloned().collect::<Vec<_>>()), // P
            a_R.iter()
                .chain(b_L.iter())
                .chain(std::iter::once(&c_R))
                .map(|x| x.limbs().to_vec())
                .collect(), // scalar
            base_chip.limb_bits, // max bits
            var_window_bits,     // window bits
        );

        L_vec.push(L);
        R_vec.push(R);

        // transcript.append_point(b"L", &L);
        // transcript.append_point(b"R", &R);

        // let u = transcript.challenge_scalar(b"u");

        let u_valid: Vec<Reduced<ProperCrtUint<F>, SF>> =
        u.iter().map(|u_i| scalar_chip.enforce_less_than(ctx, (*u_i).clone())).collect();

        let SF_ONE = scalar_chip.load_constant(ctx, SF::one());

        let u_inv: Vec<_> = u_valid
            .iter()
            .map(|u_i| scalar_chip.divide(ctx, SF_ONE.clone(), ProperCrtUint::from((*u_i).clone())))
            .collect();

        // let u_inv = u.invert();
        
        for i in 0..n {
            a_L[i] = scalar_chip.add_no_carry(ctx, scalar_chip.mul(ctx, a_L[i].clone(), u_valid[i].clone()), scalar_chip.mul(ctx, u_inv[i].clone(), a_R[i].clone()));
            b_L[i] = scalar_chip.add_no_carry(ctx, scalar_chip.mul(ctx, b_L[i].clone(), u_inv[i].clone()), scalar_chip.mul(ctx, u_valid[i].clone(), b_R[i].clone()));

            G_L[i] = multi_scalar_multiply::<_, _, GA>(
                base_chip,
                ctx,
                &[G_L[i], G_R[i]], // P
                &[u_inv, u], // scalar
                base_chip.limb_bits, // max bits
                var_window_bits,     // window bits
            );


            H_L[i] = multi_scalar_multiply::<_, _, GA>(
                base_chip,
                ctx,
                &[H_L[i], H_R[i]], // P
                &[u_inv, u], // scalar
                base_chip.limb_bits, // max bits
                var_window_bits,     // window bits
            );
        }

        a_valid = a_L;
        b_valid = b_L;
        G = G_L;
        H = H_L;

    }

    // todo: implement while loop
    // - differences: g_factor and h_factor array + diff in the ristretto points
    // todo: return

}

/// Computes three vectors of verification scalars \\([u\_{i}^{2}]\\), \\([u\_{i}^{-2}]\\) and \\([s\_{i}]\\) for combined multiscalar multiplication
/// u_{i} is provided as input, assume is checked to be in [0, n - 1]
/// returns (u_{i}^{2}, u_{i}^{-2}, s_{i}) with size k, k, 2^k
///
/// todo: desc: takes in a vector of u_is
/// todo: q: what are the field stuff
pub fn verification_scalars<'range, F: PrimeField, SF: PrimeField>(
    ctx: &mut Context<F>,
    // todo: desc: in the bulletproofs impl, the u vec is stored in self
    u: Vec<Reduced<ProperCrtUint<F>, SF>>, // size = k, u_i < n, randomness generated by fiat-shamir transform
    scalar_chip: &FpChip<'range, F, SF>,
    // todo: desc: uses ProperCrtUint<F> instead of Scalar like the bulletproof impl
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

    let SF_ONE = scalar_chip.load_constant(ctx, SF::one());

    // 1
    let u_invert: Vec<_> = u
        .iter()
        .map(|u_i| scalar_chip.divide(ctx, SF_ONE.clone(), ProperCrtUint::from((*u_i).clone())))
        .collect();

    // 3
    let u_inv_pow_two: Vec<_> = u_invert
        .iter()
        .map(|u_i| {
            scalar_chip.mul(ctx, CRTInteger::from((*u_i).clone()), CRTInteger::from((*u_i).clone()))
        })
        .collect();

    // 2. Compute 1/(u_k...u_1)
    let allinv =
        u_invert.iter().fold(SF_ONE.clone(), |acc, x| scalar_chip.mul(ctx, acc, (*x).clone()));
    // compute s
    let mut s = Vec::with_capacity(2 ^ lg_n);
    s.push(allinv);
    for i in 1..2 ^ lg_n {
        let lg_i = (32 - 1 - (i as u32).leading_zeros()) as usize;
        let k = 1 << lg_i;
        let u_lg_i_sq = u_pow_two[(k - 1) - lg_i].clone();
        s.push(scalar_chip.mul(ctx, s[i - k].clone(), u_lg_i_sq));
    }

    (u_pow_two, u_inv_pow_two, s)
}

// CF is the coordinate field of GA
// SF is the scalar field of GA
// p = base field modulus
// n = scalar field modulus
/// follow verification equation in https://doc-internal.dalek.rs/bulletproofs/inner_product_proof/index.html
pub fn inner_product_proof<F: PrimeField, CF: PrimeField, SF: PrimeField, GA>(
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

    let P_prime = multi_scalar_multiply::<_, _, GA>(
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

    let is_valid_proof = chip.is_equal(ctx, P_prime, P);

    is_valid_proof
}