def rs_correct_msg(msg_in, nsym, erase_pos=None):
    if len(msg_in) > 255:
        raise ValueError("Message is too long (%i when max is 255)" % len(msg_in))
    msg_out = list(msg_in)
    if erase_pos is None:
        erase_pos = []
    else:
        for e_pos in erase_pos:
            msg_out[e_pos] = 0
    if len(erase_pos) > nsym: raise ReedSolomonError("Too many erasures to correct")
    synd = rs_calc_syndromes(msg_out, nsym)
    if max(synd) == 0:
        return msg_out[:-nsym], msg_out[-nsym:]
    fsynd = rs_forney_syndromes(synd, erase_pos, len(msg_out))
    err_loc = rs_find_error_locator(fsynd, nsym, erase_count=len(erase_pos))
    err_pos = rs_find_errors(err_loc[::-1] , len(msg_out))
    if err_pos is None:
        raise ReedSolomonError("Could not locate error")
    msg_out = rs_correct_errata(msg_out, synd, (erase_pos + err_pos))
    synd = rs_calc_syndromes(msg_out, nsym)
    if max(synd) > 0:
        raise ReedSolomonError("Could not correct message")
    return msg_out[:-nsym], msg_out[-nsym:]



def rs_forney_syndromes(synd, pos, nmess):
    erase_pos_reversed = [nmess-1-p for p in pos]
    fsynd = list(synd[1:])
    for i in range(0, len(pos)):
        x = gf_pow(2, erase_pos_reversed[i])
        for j in range(0, len(fsynd) - 1):
            fsynd[j] = gf_mul(fsynd[j], x) ^ fsynd[j + 1]
    return fsynd



def rs_find_errors(err_loc, nmess):
    errs = len(err_loc) - 1
    err_pos = []
    for i in range(nmess):
        if gf_poly_eval(err_loc, gf_pow(2, i)) == 0:
            err_pos.append(nmess - 1 - i)
    if len(err_pos) != errs:
        raise ReedSolomonError("Too many (or few) errors found by Chien Search for the errata locator polynomial!")
    return err_pos



def rs_find_error_locator(synd, nsym, erase_loc=None, erase_count=0):
    if erase_loc:
        err_loc = list(erase_loc)
        old_loc = list(erase_loc)
    else:
        err_loc = [1]
        old_loc = [1]
    synd_shift = len(synd) - nsym
    for i in range(0, nsym-erase_count):
        if erase_loc:
            K = erase_count+i+synd_shift
        else:
            K = i+synd_shift
        delta = synd[K]
        for j in range(1, len(err_loc)):
            delta ^= gf_mul(err_loc[-(j+1)], synd[K - j])
        old_loc = old_loc + [0]
        if delta != 0:
            if len(old_loc) > len(err_loc):
                new_loc = gf_poly_scale(old_loc, delta)
                old_loc = gf_poly_scale(err_loc, gf_inverse(delta))
                err_loc = new_loc
            err_loc = gf_poly_add(err_loc, gf_poly_scale(old_loc, delta))
    while len(err_loc) and err_loc[0] == 0: del err_loc[0]
    errs = len(err_loc) - 1
    if (errs-erase_count) * 2 + erase_count > nsym:
        raise ReedSolomonError("Too many errors to correct")
    return err_loc



def rs_correct_errata(msg_in, synd, err_pos):
    coef_pos = [len(msg_in) - 1 - p for p in err_pos]
    err_loc = rs_find_errata_locator(coef_pos)
    err_eval = rs_find_error_evaluator(synd[::-1], err_loc, len(err_loc)-1)[::-1]
    X = []
    for i in range(0, len(coef_pos)):
        l = 255 - coef_pos[i]
        X.append( gf_pow(2, -l) )
    E = [0] * (len(msg_in))
    Xlength = len(X)
    for i, Xi in enumerate(X):
        Xi_inv = gf_inverse(Xi)
        err_loc_prime_tmp = []
        for j in range(0, Xlength):
            if j != i:
                err_loc_prime_tmp.append( gf_sub(1, gf_mul(Xi_inv, X[j])) )
        err_loc_prime = 1
        for coef in err_loc_prime_tmp:
            err_loc_prime = gf_mul(err_loc_prime, coef)
        y = gf_poly_eval(err_eval[::-1], Xi_inv)
        y = gf_mul(gf_pow(Xi, 1), y)
        if err_loc_prime == 0:
            raise ReedSolomonError("Could not find error magnitude")
        magnitude = gf_div(y, err_loc_prime)
        E[err_pos[i]] = magnitude
    msg_in = gf_poly_add(msg_in, E)
    return msg_in



def rs_find_error_evaluator(synd, err_loc, nsym):
    _, remainder = gf_poly_div( gf_poly_mul(synd, err_loc), ([1] + [0]*(nsym+1)) )
    return remainder



def rs_find_errata_locator(e_pos):
    e_loc = [1]
    for i in e_pos:
        e_loc = gf_poly_mul( e_loc, gf_poly_add([1], [gf_pow(2, i), 0]) )
    return e_loc



def rs_check(msg, nsym):
    return ( max(rs_calc_syndromes(msg, nsym)) == 0 )



def rs_calc_syndromes(msg, nsym):
    synd = [0] * nsym
    for i in range(0, nsym):
        synd[i] = gf_poly_eval(msg, gf_pow(2,i))
    return [0] + synd



def rs_encode_msg(msg_in, nsym):
    if (len(msg_in) + nsym) > 255: raise ValueError("Message is too long (%i when max is 255)" % (len(msg_in)+nsym))
    gen = rs_generator_poly(nsym)
    msg_out = [0] * (len(msg_in) + len(gen)-1)
    msg_out[:len(msg_in)] = msg_in
    for i in range(len(msg_in)):
        coef = msg_out[i]
        if coef != 0:
            for j in range(1, len(gen)):
                msg_out[i+j] ^= gf_mul(gen[j], coef)
    msg_out[:len(msg_in)] = msg_in

    return msg_out



def gf_poly_div(dividend, divisor):
    msg_out = list(dividend)
    for i in range(0, len(dividend) - (len(divisor)-1)):
        coef = msg_out[i]
        if coef != 0:
            for j in range(1, len(divisor)):
                if divisor[j] != 0:
                    msg_out[i + j] ^= gf_mul(divisor[j], coef)
    separator = -(len(divisor)-1)
    return msg_out[:separator], msg_out[separator:]



def rs_generator_poly(nsym):
    g = [1]
    for i in range(0, nsym):
        g = gf_poly_mul(g, [1, gf_pow(2, i)])
    return g


    
class ReedSolomonError(Exception):
    pass



def gf_poly_eval(poly, x):
    y = poly[0]
    for i in range(1, len(poly)):
        y = gf_mul(y, x) ^ poly[i]
    return y



def gf_poly_mul(p,q):
    r = [0] * (len(p)+len(q)-1)
    for j in range(0, len(q)):
        for i in range(0, len(p)):
            r[i+j] ^= gf_mul(p[i], q[j])
    return r



def gf_poly_add(p,q):
    r = [0] * max(len(p),len(q))
    for i in range(0,len(p)):
        r[i+len(r)-len(p)] = p[i]
    for i in range(0,len(q)):
        r[i+len(r)-len(q)] ^= q[i]
    return r



def gf_poly_scale(p,x):
    r = [0] * len(p)
    for i in range(0, len(p)):
        r[i] = gf_mul(p[i], x)
    return r



def gf_pow(x, power):
    return gf_exp[(gf_log[x] * power) % 255]



def gf_inverse(x):
    return gf_exp[255 - gf_log[x]]



def gf_div(x,y):
    if y==0:
        raise ZeroDivisionError()
    if x==0:
        return 0
    return gf_exp[(gf_log[x] + 255 - gf_log[y]) % 255]



def gf_mul(x,y):
    if x==0 or y==0:
        return 0
    return gf_exp[gf_log[x] + gf_log[y]]



def init_tables(prim=0x11d):
    global gf_exp, gf_log
    gf_exp = [0] * 512
    gf_log = [0] * 256
    x = 1
    for i in range(0, 255):
        gf_exp[i] = x
        gf_log[x] = i
        x = gf_mult_noLUT(x, 2, prim)
    for i in range(255, 512):
        gf_exp[i] = gf_exp[i - 255]
    return [gf_log, gf_exp]
#init_tables()



def gf_mult_noLUT(x, y, prim=0, field_charac_full=256, carryless=True):
    r = 0
    while y:
        if y & 1: r = r ^ x if carryless else r + x
        y = y >> 1
        x = x << 1
        if prim > 0 and x & field_charac_full: x = x ^ prim
    return r


def cl_mul(x,y):
    z = 0
    i = 0
    while (y>>i) > 0:
        if y & (1<<i):
            z ^= x<<i
        i += 1
    return z



def gf_add(x, y):
    return x ^ y



def gf_sub(x, y):
    return x ^ y