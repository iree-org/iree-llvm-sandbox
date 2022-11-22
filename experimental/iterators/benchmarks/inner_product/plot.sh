#!/usr/bin/env bash

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

PLOT_SCRIPT="${SOURCE_DIR}/plot.py"

#
# Parse command line parameters
#
print_usage() {
  echo "Usage: $0 [-i INFILE] [-o OUTDIR]" 1>&2
  exit 1
}

infile="${SOURCE_DIR}/result.jsonl"
outdir="${SOURCE_DIR}/"

# Parse options.
while getopts ":i:o:" o; do
  case "${o}" in
    i)
      infile=${OPTARG}
      ;;
    o)
      outdir=${OPTARG}
      ;;
    *)
      print_usage
      ;;
  esac
done
shift $((OPTIND-1))

if [ "$#" -ne 0 ]; then
  print_usage
fi

p=time_by_method_dtype; "$PLOT_SCRIPT" $p -i "$infile" -o "$outdir/run_${p}.pdf" -p run
p=time_by_method_dtype; "$PLOT_SCRIPT" $p -i "$infile" -o "$outdir/compile_${p}.pdf" -p compile
p=time_by_num_elements; "$PLOT_SCRIPT" $p -i "$infile" -o "$outdir/run_${p}.pdf" -p run
