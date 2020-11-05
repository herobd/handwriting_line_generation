import editdistance
def cer(r, h, casesensitive=True):
    if not casesensitive:
        r=r.lower()
        h=h.lower()
    #Remove any double or trailing
    r = u' '.join(r.split())
    h = u' '.join(h.split())

    return err(r, h)

def err(r, h):
    dis = editdistance.eval(r, h)
    if len(r) == 0.0:
        return len(h)

    return float(dis) / float(len(r))

def wer(r, h,  casesensitive=True):
    if not casesensitive:
        r=r.lower()
        h=h.lower()
    r = r.split()
    h = h.split()

    return err(r,h)
