"""Bounded sampler-law CLI; original training remains unchanged."""
import argparse,json,sys
from .sampler import strict_json,audit

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);a=p.parse_args(argv)
    try:
        with open(a.input,encoding='utf-8')as f:v=strict_json(f.read(50001))
        print(json.dumps(audit(v),indent=2,allow_nan=False));return 0
    except(OSError,ValueError,OverflowError)as e:print('ERROR: '+str(e),file=sys.stderr);return 2
if __name__=='__main__':sys.exit(main())
