
"""
view_results.py — Quick terminal viewer for PiF results CSV
Usage: python3 view_results.py --csv /path/to/results.csv [--n 20] [--only_success]
"""
import argparse, csv, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",         required=True)
    ap.add_argument("--n",           type=int, default=20,  help="rows to show")
    ap.add_argument("--only_success",action="store_true",   help="show only gpt=1 rows")
    ap.add_argument("--only_fail",   action="store_true",   help="show only gpt=0 rows")
    ap.add_argument("--response_len",type=int, default=300, help="response preview length")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    total = len(rows)

    # Stats
    kw  = sum(1 for r in rows if str(r.get("asr_keyword","")).lower() in ["1","true"])
    gpt = sum(1 for r in rows if str(r.get("asr_gpt","")).lower() in ["1","true"])
    det = sum(1 for r in rows if str(r.get("detailed_harmful","")).lower() in ["1","true"])
    ahs_scores = [float(r["ahs_score"]) for r in rows if r.get("ahs_score","").strip()]

    print(f"\n{'='*70}")
    print(f"FILE: {args.csv}")
    print(f"{'='*70}")
    print(f"  Total rows    : {total}")
    print(f"  ASR keyword   : {kw}/{total} = {kw/total*100:.1f}%")
    print(f"  ASR+GPT       : {gpt}/{total} = {gpt/total*100:.1f}%")
    if det: print(f"  Detailed harm : {det}/{total} = {det/total*100:.1f}%")
    if ahs_scores: print(f"  AHS avg       : {sum(ahs_scores)/len(ahs_scores):.3f} (n={len(ahs_scores)})")
    print(f"{'='*70}\n")

    # Filter
    if args.only_success:
        rows = [r for r in rows if str(r.get("asr_gpt","")).lower() in ["1","true"]]
        print(f"Showing {min(args.n, len(rows))} GPT-successful rows\n")
    elif args.only_fail:
        rows = [r for r in rows if str(r.get("asr_gpt","")).lower() not in ["1","true"]]
        print(f"Showing {min(args.n, len(rows))} GPT-failed rows\n")

    for r in rows[:args.n]:
        print(f"── [{r.get('prompt_idx','?')}] kw={r.get('asr_keyword','?')} "
              f"gpt={r.get('asr_gpt','?')} "
              f"ahs={r.get('ahs_score','?')} "
              f"det={r.get('detailed_harmful','?')} "
              f"cat={r.get('detailed_category','?')} "
              f"iters={r.get('n_iterations','?')} "
              f"time={r.get('pif_time_s','?')}s")
        print(f"   ORIG:  {r.get('original_prompt','')[:80]}")
        print(f"   JB:    {r.get('jailbreak_prompt','')[:80]}")
        print(f"   RESP:  {r.get('target_response','')[:args.response_len]}")
        if r.get("detailed_reason"):
            print(f"   WHY:   {r.get('detailed_reason','')[:120]}")
        if r.get("ahs_reason"):
            print(f"   AHS:   {r.get('ahs_reason','')[:120]}")
        print()

if __name__ == "__main__":
    main()
