#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#!/usr/bin/env python
import opentuner
from opentuner import ConfigurationManipulator
from opentuner.search.manipulator import IntegerParameter, PowerOfTwoParameter, EnumParameter, BooleanParameter
from opentuner import MeasurementInterface
from opentuner import Result
import sys

from .utils import parse

max_flops = 0
class MLIRFlagsTuner(MeasurementInterface):

  def manipulator(self):
    """
    Define the search space by creating a
    ConfigurationManipulator
    """
    manipulator = ConfigurationManipulator()

    manipulator.add_parameter(
      PowerOfTwoParameter('mr', 4, 4))

    manipulator.add_parameter(
      PowerOfTwoParameter('nr', 16, 16))

    manipulator.add_parameter(
      PowerOfTwoParameter('kr', 16, 64))

    manipulator.add_parameter(
      PowerOfTwoParameter('kc', 64, 128))

    manipulator.add_parameter(
      PowerOfTwoParameter('mc', 256, 2048))

    manipulator.add_parameter(
      PowerOfTwoParameter('nc', 64, 2048))

    manipulator.add_parameter(
      IntegerParameter('ha', 4 , 4))

    manipulator.add_parameter(
      IntegerParameter('hb', 3 , 3))

    return manipulator
  
  def run(self, desired_result, input, limit):
    global max_flops


    """
    Compile and run a given configuration then
    return performance
    """

    cfg = desired_result.configuration.data

    mr = cfg['mr']
    nr = cfg['nr']
    kr = cfg['kr']
    kc = cfg['kc']
    mc = cfg['mc']
    nc = cfg['nc']
    ha = cfg['ha']
    hb = cfg['hb']
   # reordering =  cfg['reorder']

    M = self.args.M
    N = self.args.N
    K = self.args.K

    # mr = min(mr,mc)
    # nr = min(nr,nc)
    # kr = min(kr, kc)
    # kr = kc

    cfg['mr'] = mr
    cfg['nr'] = nr
    cfg['kr'] = kr
    reordering = "Afirst"

    if  reordering == "Afirst":
      reorder_inner = "0 1 2"
      reorder_outer = "0 2 1"
    else:
      reorder_inner = "1 0 2"
      reorder_outer = "1 2 0"

    hoisting_params = f"{ha} {hb} 0"
    cmd = ['python3 -m alp.mlirc']
    cmd.append(f'--M {M}')
    cmd.append(f'--N {N}')
    cmd.append(f'--K {K}')

    cmd.append(f"--tile-sizes {mc} {nc} {kc}")
    cmd.append(f"--register-tile-sizes {mr} {nr} {kr}")
    cmd.append(f"--reorder-tile-sizes {reorder_outer}")
    cmd.append(f"--reorder-register-tile-sizes {reorder_inner}")
    
    #if cfg['unrollVectorTransfers']:
    cmd.append(f"--unroll-vector-transfers")
    cmd.append(f"--split-vector-transfers-to none") # {cfg['splitVectorTransfersTo']}")
    cmd.append(f"--hoist-packing {hoisting_params}")

    compile_result = self.call_program(' '.join(cmd))


    if compile_result['returncode'] != 0:
      return Result(time=sys.maxsize)

    assert compile_result['returncode'] == 0

    run_cmd = './exec_matmul'
    run_result = self.call_program(run_cmd, limit=0.7)

    if run_result['returncode'] != 0:
      return  Result(time=sys.maxsize)

    assert run_result['returncode'] == 0
 
    secs, flops = parse(run_result['stderr'])

    if(flops>max_flops):
      s = ' '.join([str(elem) for elem in cmd])
      max_flops=flops


    return Result(time=1/flops)

  def save_final_config(self, configuration):
    """called at the end of tuning"""
    print("Optimal block size written to mmm_final_config.json:", configuration.data)
    M = self.args.M
    N = self.args.N
    K = self.args.K
    self.manipulator().save_to_file(configuration.data,
                                    f'mmm_final_config_{M}_{N}_{K}.json')


if __name__ == '__main__':
  argparser = opentuner.default_argparser()
  argparser.add_argument("--M", type=int)
  argparser.add_argument("--N", type=int)
  argparser.add_argument("--K", type=int)
  MLIRFlagsTuner.main(argparser.parse_args())
