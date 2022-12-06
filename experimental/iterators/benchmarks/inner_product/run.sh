#!/usr/bin/env bash

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

RUN_SCRIPT="${SOURCE_DIR}/run.py"

NUM_ELEMENTS=($(for l in {8..25}; do echo $((2**l)); done))
DTYPES=(int8 int16 int32 int64 float32 float64)
METHODS=(numpy iterators)

#
# Exhaust all combinations.
#
exhaust() {(
  for n in ${NUM_ELEMENTS[@]}
  do
    # Compute number of repetitions as floor(total_num_ops/n).
    r=$((($total_num_ops + $n - 1) / $n))
    for t in ${DTYPES[@]}
    do
      for m in ${METHODS[@]}
      do
        for _ in $(seq $num_repetitions)
        do
          "$RUN_SCRIPT" -n $n -t $t -m $m -r $r
        done
      done
    done
  done | tee "$outfile"
)}

#
# Test all values of all parameters.
#
test() {(
  for n in ${NUM_ELEMENTS[@]}
  do
    "$RUN_SCRIPT" -n $n -r 2  # Run twice for stddev to make sense
  done

  for t in ${DTYPES[@]}
  do
    for m in ${METHODS[@]}
    do
      "$RUN_SCRIPT" -t $t -m $m -r 2   # Run twice for stddev to make sense
    done
  done
)}

#
# Parse command line parameters
#
print_usage() {
  echo "Usage: $0 [-o OUTFILE] [-r NUM_REPETITIONS] ACTION" 1>&2
  exit 1
}

outfile="${SOURCE_DIR}/result.jsonl"
num_repetitions=10
total_num_ops=$((2**25))

# Parse options.
while getopts ":o:r:t:" o; do
  case "${o}" in
    o)
      outfile=${OPTARG}
      ;;
    r)
      num_repetitions=${OPTARG}
      ;;
    t)
      total_num_ops=${OPTARG}
      ;;
    *)
      print_usage
      ;;
  esac
done
shift $((OPTIND-1))

# Parse action.
action=$1
shift

if [ "$#" -ne 0 ]; then
  print_usage
fi

# Call action.
case "${action}" in
  exhaust)
    exhaust
    ;;
  test)
    test
    ;;
  *)
    print_usage
    ;;
esac
