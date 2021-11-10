# pytype: skip-file

from mlir.ir import *
from mlir.dialects.linalg.opdsl.lang import *


@linalg_structured_op
def conv_1d_ncw_cfw(
    I=TensorDef(TV.T1, S.N, S.C, S.OW * S.SW + S.KW * S.DW),
    K=TensorDef(TV.T2, S.C, S.F, S.KW),
    O=TensorDef(U, S.N, S.F, S.OW, output=True),
    strides=AttributeDef(S.SW),
    dilations=AttributeDef(S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.f, D.ow, D.c, D.kw)
  O[D.n, D.f, D.ow] += (
      cast(U, I[D.n, D.c, D.ow * S.SW + D.kw * S.DW]) *
      cast(U, K[D.c, D.f, D.kw]))


@linalg_structured_op
def conv_1d_ncw_cwf(
    I=TensorDef(TV.T1, S.N, S.C, S.OW * S.SW + S.KW * S.DW),
    K=TensorDef(TV.T2, S.C, S.KW, S.F),
    O=TensorDef(U, S.N, S.OW, S.F, output=True),
    strides=AttributeDef(S.SW),
    dilations=AttributeDef(S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.ow, D.f, D.c, D.kw)
  O[D.n, D.ow, D.f] += (
      cast(U, I[D.n, D.c, D.ow * S.SW + D.kw * S.DW]) *
      cast(U, K[D.c, D.kw, D.f]))


@linalg_structured_op
def conv_1d_ncw_fcw(
    I=TensorDef(TV.T1, S.N, S.C, S.OW * S.SW + S.KW * S.DW),
    K=TensorDef(TV.T2, S.F, S.C, S.KW),
    O=TensorDef(U, S.N, S.F, S.OW, output=True),
    strides=AttributeDef(S.SW),
    dilations=AttributeDef(S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.f, D.ow, D.c, D.kw)
  O[D.n, D.f, D.ow] += (
      cast(U, I[D.n, D.c, D.ow * S.SW + D.kw * S.DW]) *
      cast(U, K[D.f, D.c, D.kw]))


@linalg_structured_op
def conv_1d_ncw_fwc(
    I=TensorDef(TV.T1, S.N, S.C, S.OW * S.SW + S.KW * S.DW),
    K=TensorDef(TV.T2, S.F, S.KW, S.C),
    O=TensorDef(U, S.N, S.F, S.OW, output=True),
    strides=AttributeDef(S.SW),
    dilations=AttributeDef(S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.f, D.ow, D.kw, D.c)
  O[D.n, D.f, D.ow] += (
      cast(U, I[D.n, D.c, D.ow * S.SW + D.kw * S.DW]) *
      cast(U, K[D.f, D.kw, D.c]))


@linalg_structured_op
def conv_1d_ncw_wcf(
    I=TensorDef(TV.T1, S.N, S.C, S.OW * S.SW + S.KW * S.DW),
    K=TensorDef(TV.T2, S.KW, S.C, S.F),
    O=TensorDef(U, S.N, S.OW, S.F, output=True),
    strides=AttributeDef(S.SW),
    dilations=AttributeDef(S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.ow, D.f, D.kw, D.c)
  O[D.n, D.ow, D.f] += (
      cast(U, I[D.n, D.c, D.ow * S.SW + D.kw * S.DW]) *
      cast(U, K[D.kw, D.c, D.f]))


@linalg_structured_op
def conv_1d_ncw_wfc(
    I=TensorDef(TV.T1, S.N, S.C, S.OW * S.SW + S.KW * S.DW),
    K=TensorDef(TV.T2, S.KW, S.F, S.C),
    O=TensorDef(U, S.N, S.OW, S.F, output=True),
    strides=AttributeDef(S.SW),
    dilations=AttributeDef(S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.ow, D.f, D.kw, D.c)
  O[D.n, D.ow, D.f] += (
      cast(U, I[D.n, D.c, D.ow * S.SW + D.kw * S.DW]) *
      cast(U, K[D.kw, D.f, D.c]))


@linalg_structured_op
def conv_1d_nwc_cfw(
    I=TensorDef(TV.T1, S.N, S.OW * S.SW + S.KW * S.DW, S.C),
    K=TensorDef(TV.T2, S.C, S.F, S.KW),
    O=TensorDef(U, S.N, S.F, S.OW, output=True),
    strides=AttributeDef(S.SW),
    dilations=AttributeDef(S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.f, D.ow, D.c, D.kw)
  O[D.n, D.f, D.ow] += (
      cast(U, I[D.n, D.ow * S.SW + D.kw * S.DW, D.c]) *
      cast(U, K[D.c, D.f, D.kw]))


@linalg_structured_op
def conv_1d_nwc_cwf(
    I=TensorDef(TV.T1, S.N, S.OW * S.SW + S.KW * S.DW, S.C),
    K=TensorDef(TV.T2, S.C, S.KW, S.F),
    O=TensorDef(U, S.N, S.OW, S.F, output=True),
    strides=AttributeDef(S.SW),
    dilations=AttributeDef(S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.ow, D.f, D.c, D.kw)
  O[D.n, D.ow, D.f] += (
      cast(U, I[D.n, D.ow * S.SW + D.kw * S.DW, D.c]) *
      cast(U, K[D.c, D.kw, D.f]))


@linalg_structured_op
def conv_1d_nwc_fcw(
    I=TensorDef(TV.T1, S.N, S.OW * S.SW + S.KW * S.DW, S.C),
    K=TensorDef(TV.T2, S.F, S.C, S.KW),
    O=TensorDef(U, S.N, S.F, S.OW, output=True),
    strides=AttributeDef(S.SW),
    dilations=AttributeDef(S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.f, D.ow, D.c, D.kw)
  O[D.n, D.f, D.ow] += (
      cast(U, I[D.n, D.ow * S.SW + D.kw * S.DW, D.c]) *
      cast(U, K[D.f, D.c, D.kw]))


@linalg_structured_op
def conv_1d_nwc_fwc(
    I=TensorDef(TV.T1, S.N, S.OW * S.SW + S.KW * S.DW, S.C),
    K=TensorDef(TV.T2, S.F, S.KW, S.C),
    O=TensorDef(U, S.N, S.F, S.OW, output=True),
    strides=AttributeDef(S.SW),
    dilations=AttributeDef(S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.f, D.ow, D.kw, D.c)
  O[D.n, D.f, D.ow] += (
      cast(U, I[D.n, D.ow * S.SW + D.kw * S.DW, D.c]) *
      cast(U, K[D.f, D.kw, D.c]))


@linalg_structured_op
def conv_1d_nwc_wcf(
    I=TensorDef(TV.T1, S.N, S.OW * S.SW + S.KW * S.DW, S.C),
    K=TensorDef(TV.T2, S.KW, S.C, S.F),
    O=TensorDef(U, S.N, S.OW, S.F, output=True),
    strides=AttributeDef(S.SW),
    dilations=AttributeDef(S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.ow, D.f, D.kw, D.c)
  O[D.n, D.ow, D.f] += (
      cast(U, I[D.n, D.ow * S.SW + D.kw * S.DW, D.c]) *
      cast(U, K[D.kw, D.c, D.f]))


@linalg_structured_op
def conv_1d_nwc_wfc(
    I=TensorDef(TV.T1, S.N, S.OW * S.SW + S.KW * S.DW, S.C),
    K=TensorDef(TV.T2, S.KW, S.F, S.C),
    O=TensorDef(U, S.N, S.OW, S.F, output=True),
    strides=AttributeDef(S.SW),
    dilations=AttributeDef(S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.ow, D.f, D.kw, D.c)
  O[D.n, D.ow, D.f] += (
      cast(U, I[D.n, D.ow * S.SW + D.kw * S.DW, D.c]) *
      cast(U, K[D.kw, D.f, D.c]))


@linalg_structured_op
def conv_1d_cnw_cfw(
    I=TensorDef(TV.T1, S.C, S.N, S.OW * S.SW + S.KW * S.DW),
    K=TensorDef(TV.T2, S.C, S.F, S.KW),
    O=TensorDef(U, S.N, S.F, S.OW, output=True),
    strides=AttributeDef(S.SW),
    dilations=AttributeDef(S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.f, D.ow, D.c, D.kw)
  O[D.n, D.f, D.ow] += (
      cast(U, I[D.c, D.n, D.ow * S.SW + D.kw * S.DW]) *
      cast(U, K[D.c, D.f, D.kw]))


@linalg_structured_op
def conv_1d_cnw_cwf(
    I=TensorDef(TV.T1, S.C, S.N, S.OW * S.SW + S.KW * S.DW),
    K=TensorDef(TV.T2, S.C, S.KW, S.F),
    O=TensorDef(U, S.N, S.OW, S.F, output=True),
    strides=AttributeDef(S.SW),
    dilations=AttributeDef(S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.ow, D.f, D.c, D.kw)
  O[D.n, D.ow, D.f] += (
      cast(U, I[D.c, D.n, D.ow * S.SW + D.kw * S.DW]) *
      cast(U, K[D.c, D.kw, D.f]))


@linalg_structured_op
def conv_1d_cnw_fcw(
    I=TensorDef(TV.T1, S.C, S.N, S.OW * S.SW + S.KW * S.DW),
    K=TensorDef(TV.T2, S.F, S.C, S.KW),
    O=TensorDef(U, S.N, S.F, S.OW, output=True),
    strides=AttributeDef(S.SW),
    dilations=AttributeDef(S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.f, D.ow, D.c, D.kw)
  O[D.n, D.f, D.ow] += (
      cast(U, I[D.c, D.n, D.ow * S.SW + D.kw * S.DW]) *
      cast(U, K[D.f, D.c, D.kw]))


@linalg_structured_op
def conv_1d_cnw_fwc(
    I=TensorDef(TV.T1, S.C, S.N, S.OW * S.SW + S.KW * S.DW),
    K=TensorDef(TV.T2, S.F, S.KW, S.C),
    O=TensorDef(U, S.N, S.F, S.OW, output=True),
    strides=AttributeDef(S.SW),
    dilations=AttributeDef(S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.f, D.ow, D.kw, D.c)
  O[D.n, D.f, D.ow] += (
      cast(U, I[D.c, D.n, D.ow * S.SW + D.kw * S.DW]) *
      cast(U, K[D.f, D.kw, D.c]))


@linalg_structured_op
def conv_1d_cnw_wcf(
    I=TensorDef(TV.T1, S.C, S.N, S.OW * S.SW + S.KW * S.DW),
    K=TensorDef(TV.T2, S.KW, S.C, S.F),
    O=TensorDef(U, S.N, S.OW, S.F, output=True),
    strides=AttributeDef(S.SW),
    dilations=AttributeDef(S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.ow, D.f, D.kw, D.c)
  O[D.n, D.ow, D.f] += (
      cast(U, I[D.c, D.n, D.ow * S.SW + D.kw * S.DW]) *
      cast(U, K[D.kw, D.c, D.f]))


@linalg_structured_op
def conv_1d_cnw_wfc(
    I=TensorDef(TV.T1, S.C, S.N, S.OW * S.SW + S.KW * S.DW),
    K=TensorDef(TV.T2, S.KW, S.F, S.C),
    O=TensorDef(U, S.N, S.OW, S.F, output=True),
    strides=AttributeDef(S.SW),
    dilations=AttributeDef(S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.ow, D.f, D.kw, D.c)
  O[D.n, D.ow, D.f] += (
      cast(U, I[D.c, D.n, D.ow * S.SW + D.kw * S.DW]) *
      cast(U, K[D.kw, D.f, D.c]))


@linalg_structured_op
def conv_1d_cwn_cfw(
    I=TensorDef(TV.T1, S.C, S.OW * S.SW + S.KW * S.DW, S.N),
    K=TensorDef(TV.T2, S.C, S.F, S.KW),
    O=TensorDef(U, S.F, S.OW, S.N, output=True),
    strides=AttributeDef(S.SW),
    dilations=AttributeDef(S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.f, D.ow, D.n, D.c, D.kw)
  O[D.f, D.ow, D.n] += (
      cast(U, I[D.c, D.ow * S.SW + D.kw * S.DW, D.n]) *
      cast(U, K[D.c, D.f, D.kw]))


@linalg_structured_op
def conv_1d_cwn_cwf(
    I=TensorDef(TV.T1, S.C, S.OW * S.SW + S.KW * S.DW, S.N),
    K=TensorDef(TV.T2, S.C, S.KW, S.F),
    O=TensorDef(U, S.OW, S.N, S.F, output=True),
    strides=AttributeDef(S.SW),
    dilations=AttributeDef(S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.ow, D.n, D.f, D.c, D.kw)
  O[D.ow, D.n, D.f] += (
      cast(U, I[D.c, D.ow * S.SW + D.kw * S.DW, D.n]) *
      cast(U, K[D.c, D.kw, D.f]))


@linalg_structured_op
def conv_1d_cwn_fcw(
    I=TensorDef(TV.T1, S.C, S.OW * S.SW + S.KW * S.DW, S.N),
    K=TensorDef(TV.T2, S.F, S.C, S.KW),
    O=TensorDef(U, S.F, S.OW, S.N, output=True),
    strides=AttributeDef(S.SW),
    dilations=AttributeDef(S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.f, D.ow, D.n, D.c, D.kw)
  O[D.f, D.ow, D.n] += (
      cast(U, I[D.c, D.ow * S.SW + D.kw * S.DW, D.n]) *
      cast(U, K[D.f, D.c, D.kw]))


@linalg_structured_op
def conv_1d_cwn_fwc(
    I=TensorDef(TV.T1, S.C, S.OW * S.SW + S.KW * S.DW, S.N),
    K=TensorDef(TV.T2, S.F, S.KW, S.C),
    O=TensorDef(U, S.F, S.OW, S.N, output=True),
    strides=AttributeDef(S.SW),
    dilations=AttributeDef(S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.f, D.ow, D.n, D.kw, D.c)
  O[D.f, D.ow, D.n] += (
      cast(U, I[D.c, D.ow * S.SW + D.kw * S.DW, D.n]) *
      cast(U, K[D.f, D.kw, D.c]))


@linalg_structured_op
def conv_1d_cwn_wcf(
    I=TensorDef(TV.T1, S.C, S.OW * S.SW + S.KW * S.DW, S.N),
    K=TensorDef(TV.T2, S.KW, S.C, S.F),
    O=TensorDef(U, S.OW, S.N, S.F, output=True),
    strides=AttributeDef(S.SW),
    dilations=AttributeDef(S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.ow, D.n, D.f, D.kw, D.c)
  O[D.ow, D.n, D.f] += (
      cast(U, I[D.c, D.ow * S.SW + D.kw * S.DW, D.n]) *
      cast(U, K[D.kw, D.c, D.f]))


@linalg_structured_op
def conv_1d_cwn_wfc(
    I=TensorDef(TV.T1, S.C, S.OW * S.SW + S.KW * S.DW, S.N),
    K=TensorDef(TV.T2, S.KW, S.F, S.C),
    O=TensorDef(U, S.OW, S.N, S.F, output=True),
    strides=AttributeDef(S.SW),
    dilations=AttributeDef(S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.ow, D.n, D.f, D.kw, D.c)
  O[D.ow, D.n, D.f] += (
      cast(U, I[D.c, D.ow * S.SW + D.kw * S.DW, D.n]) *
      cast(U, K[D.kw, D.f, D.c]))


@linalg_structured_op
def conv_1d_wnc_cfw(
    I=TensorDef(TV.T1, S.OW * S.SW + S.KW * S.DW, S.N, S.C),
    K=TensorDef(TV.T2, S.C, S.F, S.KW),
    O=TensorDef(U, S.F, S.OW, S.N, output=True),
    strides=AttributeDef(S.SW),
    dilations=AttributeDef(S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.f, D.ow, D.n, D.c, D.kw)
  O[D.f, D.ow, D.n] += (
      cast(U, I[D.ow * S.SW + D.kw * S.DW, D.n, D.c]) *
      cast(U, K[D.c, D.f, D.kw]))


@linalg_structured_op
def conv_1d_wnc_cwf(
    I=TensorDef(TV.T1, S.OW * S.SW + S.KW * S.DW, S.N, S.C),
    K=TensorDef(TV.T2, S.C, S.KW, S.F),
    O=TensorDef(U, S.OW, S.N, S.F, output=True),
    strides=AttributeDef(S.SW),
    dilations=AttributeDef(S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.ow, D.n, D.f, D.c, D.kw)
  O[D.ow, D.n, D.f] += (
      cast(U, I[D.ow * S.SW + D.kw * S.DW, D.n, D.c]) *
      cast(U, K[D.c, D.kw, D.f]))


@linalg_structured_op
def conv_1d_wnc_fcw(
    I=TensorDef(TV.T1, S.OW * S.SW + S.KW * S.DW, S.N, S.C),
    K=TensorDef(TV.T2, S.F, S.C, S.KW),
    O=TensorDef(U, S.F, S.OW, S.N, output=True),
    strides=AttributeDef(S.SW),
    dilations=AttributeDef(S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.f, D.ow, D.n, D.c, D.kw)
  O[D.f, D.ow, D.n] += (
      cast(U, I[D.ow * S.SW + D.kw * S.DW, D.n, D.c]) *
      cast(U, K[D.f, D.c, D.kw]))


@linalg_structured_op
def conv_1d_wnc_fwc(
    I=TensorDef(TV.T1, S.OW * S.SW + S.KW * S.DW, S.N, S.C),
    K=TensorDef(TV.T2, S.F, S.KW, S.C),
    O=TensorDef(U, S.F, S.OW, S.N, output=True),
    strides=AttributeDef(S.SW),
    dilations=AttributeDef(S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.f, D.ow, D.n, D.kw, D.c)
  O[D.f, D.ow, D.n] += (
      cast(U, I[D.ow * S.SW + D.kw * S.DW, D.n, D.c]) *
      cast(U, K[D.f, D.kw, D.c]))


@linalg_structured_op
def conv_1d_wnc_wcf(
    I=TensorDef(TV.T1, S.OW * S.SW + S.KW * S.DW, S.N, S.C),
    K=TensorDef(TV.T2, S.KW, S.C, S.F),
    O=TensorDef(U, S.OW, S.N, S.F, output=True),
    strides=AttributeDef(S.SW),
    dilations=AttributeDef(S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.ow, D.n, D.f, D.kw, D.c)
  O[D.ow, D.n, D.f] += (
      cast(U, I[D.ow * S.SW + D.kw * S.DW, D.n, D.c]) *
      cast(U, K[D.kw, D.c, D.f]))


@linalg_structured_op
def conv_1d_wnc_wfc(
    I=TensorDef(TV.T1, S.OW * S.SW + S.KW * S.DW, S.N, S.C),
    K=TensorDef(TV.T2, S.KW, S.F, S.C),
    O=TensorDef(U, S.OW, S.N, S.F, output=True),
    strides=AttributeDef(S.SW),
    dilations=AttributeDef(S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.ow, D.n, D.f, D.kw, D.c)
  O[D.ow, D.n, D.f] += (
      cast(U, I[D.ow * S.SW + D.kw * S.DW, D.n, D.c]) *
      cast(U, K[D.kw, D.f, D.c]))


@linalg_structured_op
def conv_1d_wcn_cfw(
    I=TensorDef(TV.T1, S.OW * S.SW + S.KW * S.DW, S.C, S.N),
    K=TensorDef(TV.T2, S.C, S.F, S.KW),
    O=TensorDef(U, S.F, S.OW, S.N, output=True),
    strides=AttributeDef(S.SW),
    dilations=AttributeDef(S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.f, D.ow, D.n, D.c, D.kw)
  O[D.f, D.ow, D.n] += (
      cast(U, I[D.ow * S.SW + D.kw * S.DW, D.c, D.n]) *
      cast(U, K[D.c, D.f, D.kw]))


@linalg_structured_op
def conv_1d_wcn_cwf(
    I=TensorDef(TV.T1, S.OW * S.SW + S.KW * S.DW, S.C, S.N),
    K=TensorDef(TV.T2, S.C, S.KW, S.F),
    O=TensorDef(U, S.OW, S.N, S.F, output=True),
    strides=AttributeDef(S.SW),
    dilations=AttributeDef(S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.ow, D.n, D.f, D.c, D.kw)
  O[D.ow, D.n, D.f] += (
      cast(U, I[D.ow * S.SW + D.kw * S.DW, D.c, D.n]) *
      cast(U, K[D.c, D.kw, D.f]))


@linalg_structured_op
def conv_1d_wcn_fcw(
    I=TensorDef(TV.T1, S.OW * S.SW + S.KW * S.DW, S.C, S.N),
    K=TensorDef(TV.T2, S.F, S.C, S.KW),
    O=TensorDef(U, S.F, S.OW, S.N, output=True),
    strides=AttributeDef(S.SW),
    dilations=AttributeDef(S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.f, D.ow, D.n, D.c, D.kw)
  O[D.f, D.ow, D.n] += (
      cast(U, I[D.ow * S.SW + D.kw * S.DW, D.c, D.n]) *
      cast(U, K[D.f, D.c, D.kw]))


@linalg_structured_op
def conv_1d_wcn_fwc(
    I=TensorDef(TV.T1, S.OW * S.SW + S.KW * S.DW, S.C, S.N),
    K=TensorDef(TV.T2, S.F, S.KW, S.C),
    O=TensorDef(U, S.F, S.OW, S.N, output=True),
    strides=AttributeDef(S.SW),
    dilations=AttributeDef(S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.f, D.ow, D.n, D.kw, D.c)
  O[D.f, D.ow, D.n] += (
      cast(U, I[D.ow * S.SW + D.kw * S.DW, D.c, D.n]) *
      cast(U, K[D.f, D.kw, D.c]))


@linalg_structured_op
def conv_1d_wcn_wcf(
    I=TensorDef(TV.T1, S.OW * S.SW + S.KW * S.DW, S.C, S.N),
    K=TensorDef(TV.T2, S.KW, S.C, S.F),
    O=TensorDef(U, S.OW, S.N, S.F, output=True),
    strides=AttributeDef(S.SW),
    dilations=AttributeDef(S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.ow, D.n, D.f, D.kw, D.c)
  O[D.ow, D.n, D.f] += (
      cast(U, I[D.ow * S.SW + D.kw * S.DW, D.c, D.n]) *
      cast(U, K[D.kw, D.c, D.f]))


@linalg_structured_op
def conv_1d_wcn_wfc(
    I=TensorDef(TV.T1, S.OW * S.SW + S.KW * S.DW, S.C, S.N),
    K=TensorDef(TV.T2, S.KW, S.F, S.C),
    O=TensorDef(U, S.OW, S.N, S.F, output=True),
    strides=AttributeDef(S.SW),
    dilations=AttributeDef(S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.ow, D.n, D.f, D.kw, D.c)
  O[D.ow, D.n, D.f] += (
      cast(U, I[D.ow * S.SW + D.kw * S.DW, D.c, D.n]) *
      cast(U, K[D.kw, D.f, D.c]))


@linalg_structured_op
def conv_2d_nchw_cfhw(
    I=TensorDef(TV.T1, S.N, S.C, S.OH * S.SH + S.KH * S.DH,
                S.OW * S.SW + S.KW * S.DW),
    K=TensorDef(TV.T2, S.C, S.F, S.KH, S.KW),
    O=TensorDef(U, S.N, S.F, S.OH, S.OW, output=True),
    strides=AttributeDef(S.SH, S.SW),
    dilations=AttributeDef(S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.f, D.oh, D.ow, D.c, D.kh, D.kw)
  O[D.n, D.f, D.oh, D.ow] += (
      cast(U, I[D.n, D.c, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW])
      * cast(U, K[D.c, D.f, D.kh, D.kw]))


@linalg_structured_op
def conv_2d_nchw_chwf(
    I=TensorDef(TV.T1, S.N, S.C, S.OH * S.SH + S.KH * S.DH,
                S.OW * S.SW + S.KW * S.DW),
    K=TensorDef(TV.T2, S.C, S.KH, S.KW, S.F),
    O=TensorDef(U, S.N, S.OH, S.OW, S.F, output=True),
    strides=AttributeDef(S.SH, S.SW),
    dilations=AttributeDef(S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.oh, D.ow, D.f, D.c, D.kh, D.kw)
  O[D.n, D.oh, D.ow, D.f] += (
      cast(U, I[D.n, D.c, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW])
      * cast(U, K[D.c, D.kh, D.kw, D.f]))


@linalg_structured_op
def conv_2d_nchw_fchw(
    I=TensorDef(TV.T1, S.N, S.C, S.OH * S.SH + S.KH * S.DH,
                S.OW * S.SW + S.KW * S.DW),
    K=TensorDef(TV.T2, S.F, S.C, S.KH, S.KW),
    O=TensorDef(U, S.N, S.F, S.OH, S.OW, output=True),
    strides=AttributeDef(S.SH, S.SW),
    dilations=AttributeDef(S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.f, D.oh, D.ow, D.c, D.kh, D.kw)
  O[D.n, D.f, D.oh, D.ow] += (
      cast(U, I[D.n, D.c, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW])
      * cast(U, K[D.f, D.c, D.kh, D.kw]))


@linalg_structured_op
def conv_2d_nchw_fhwc(
    I=TensorDef(TV.T1, S.N, S.C, S.OH * S.SH + S.KH * S.DH,
                S.OW * S.SW + S.KW * S.DW),
    K=TensorDef(TV.T2, S.F, S.KH, S.KW, S.C),
    O=TensorDef(U, S.N, S.F, S.OH, S.OW, output=True),
    strides=AttributeDef(S.SH, S.SW),
    dilations=AttributeDef(S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.f, D.oh, D.ow, D.kh, D.kw, D.c)
  O[D.n, D.f, D.oh, D.ow] += (
      cast(U, I[D.n, D.c, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW])
      * cast(U, K[D.f, D.kh, D.kw, D.c]))


@linalg_structured_op
def conv_2d_nchw_hwcf(
    I=TensorDef(TV.T1, S.N, S.C, S.OH * S.SH + S.KH * S.DH,
                S.OW * S.SW + S.KW * S.DW),
    K=TensorDef(TV.T2, S.KH, S.KW, S.C, S.F),
    O=TensorDef(U, S.N, S.OH, S.OW, S.F, output=True),
    strides=AttributeDef(S.SH, S.SW),
    dilations=AttributeDef(S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.oh, D.ow, D.f, D.kh, D.kw, D.c)
  O[D.n, D.oh, D.ow, D.f] += (
      cast(U, I[D.n, D.c, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW])
      * cast(U, K[D.kh, D.kw, D.c, D.f]))


@linalg_structured_op
def conv_2d_nchw_hwfc(
    I=TensorDef(TV.T1, S.N, S.C, S.OH * S.SH + S.KH * S.DH,
                S.OW * S.SW + S.KW * S.DW),
    K=TensorDef(TV.T2, S.KH, S.KW, S.F, S.C),
    O=TensorDef(U, S.N, S.OH, S.OW, S.F, output=True),
    strides=AttributeDef(S.SH, S.SW),
    dilations=AttributeDef(S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.oh, D.ow, D.f, D.kh, D.kw, D.c)
  O[D.n, D.oh, D.ow, D.f] += (
      cast(U, I[D.n, D.c, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW])
      * cast(U, K[D.kh, D.kw, D.f, D.c]))


@linalg_structured_op
def conv_2d_nhwc_cfhw(
    I=TensorDef(TV.T1, S.N, S.OH * S.SH + S.KH * S.DH,
                S.OW * S.SW + S.KW * S.DW, S.C),
    K=TensorDef(TV.T2, S.C, S.F, S.KH, S.KW),
    O=TensorDef(U, S.N, S.F, S.OH, S.OW, output=True),
    strides=AttributeDef(S.SH, S.SW),
    dilations=AttributeDef(S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.f, D.oh, D.ow, D.c, D.kh, D.kw)
  O[D.n, D.f, D.oh, D.ow] += (
      cast(U, I[D.n, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW, D.c])
      * cast(U, K[D.c, D.f, D.kh, D.kw]))


@linalg_structured_op
def conv_2d_nhwc_chwf(
    I=TensorDef(TV.T1, S.N, S.OH * S.SH + S.KH * S.DH,
                S.OW * S.SW + S.KW * S.DW, S.C),
    K=TensorDef(TV.T2, S.C, S.KH, S.KW, S.F),
    O=TensorDef(U, S.N, S.OH, S.OW, S.F, output=True),
    strides=AttributeDef(S.SH, S.SW),
    dilations=AttributeDef(S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.oh, D.ow, D.f, D.c, D.kh, D.kw)
  O[D.n, D.oh, D.ow, D.f] += (
      cast(U, I[D.n, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW, D.c])
      * cast(U, K[D.c, D.kh, D.kw, D.f]))


@linalg_structured_op
def conv_2d_nhwc_fchw(
    I=TensorDef(TV.T1, S.N, S.OH * S.SH + S.KH * S.DH,
                S.OW * S.SW + S.KW * S.DW, S.C),
    K=TensorDef(TV.T2, S.F, S.C, S.KH, S.KW),
    O=TensorDef(U, S.N, S.F, S.OH, S.OW, output=True),
    strides=AttributeDef(S.SH, S.SW),
    dilations=AttributeDef(S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.f, D.oh, D.ow, D.c, D.kh, D.kw)
  O[D.n, D.f, D.oh, D.ow] += (
      cast(U, I[D.n, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW, D.c])
      * cast(U, K[D.f, D.c, D.kh, D.kw]))


@linalg_structured_op
def conv_2d_nhwc_fhwc(
    I=TensorDef(TV.T1, S.N, S.OH * S.SH + S.KH * S.DH,
                S.OW * S.SW + S.KW * S.DW, S.C),
    K=TensorDef(TV.T2, S.F, S.KH, S.KW, S.C),
    O=TensorDef(U, S.N, S.F, S.OH, S.OW, output=True),
    strides=AttributeDef(S.SH, S.SW),
    dilations=AttributeDef(S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.f, D.oh, D.ow, D.kh, D.kw, D.c)
  O[D.n, D.f, D.oh, D.ow] += (
      cast(U, I[D.n, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW, D.c])
      * cast(U, K[D.f, D.kh, D.kw, D.c]))


@linalg_structured_op
def conv_2d_nhwc_hwcf(
    I=TensorDef(TV.T1, S.N, S.OH * S.SH + S.KH * S.DH,
                S.OW * S.SW + S.KW * S.DW, S.C),
    K=TensorDef(TV.T2, S.KH, S.KW, S.C, S.F),
    O=TensorDef(U, S.N, S.OH, S.OW, S.F, output=True),
    strides=AttributeDef(S.SH, S.SW),
    dilations=AttributeDef(S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.oh, D.ow, D.f, D.kh, D.kw, D.c)
  O[D.n, D.oh, D.ow, D.f] += (
      cast(U, I[D.n, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW, D.c])
      * cast(U, K[D.kh, D.kw, D.c, D.f]))


@linalg_structured_op
def conv_2d_nhwc_hwfc(
    I=TensorDef(TV.T1, S.N, S.OH * S.SH + S.KH * S.DH,
                S.OW * S.SW + S.KW * S.DW, S.C),
    K=TensorDef(TV.T2, S.KH, S.KW, S.F, S.C),
    O=TensorDef(U, S.N, S.OH, S.OW, S.F, output=True),
    strides=AttributeDef(S.SH, S.SW),
    dilations=AttributeDef(S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.oh, D.ow, D.f, D.kh, D.kw, D.c)
  O[D.n, D.oh, D.ow, D.f] += (
      cast(U, I[D.n, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW, D.c])
      * cast(U, K[D.kh, D.kw, D.f, D.c]))


@linalg_structured_op
def conv_2d_cnhw_cfhw(
    I=TensorDef(TV.T1, S.C, S.N, S.OH * S.SH + S.KH * S.DH,
                S.OW * S.SW + S.KW * S.DW),
    K=TensorDef(TV.T2, S.C, S.F, S.KH, S.KW),
    O=TensorDef(U, S.N, S.F, S.OH, S.OW, output=True),
    strides=AttributeDef(S.SH, S.SW),
    dilations=AttributeDef(S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.f, D.oh, D.ow, D.c, D.kh, D.kw)
  O[D.n, D.f, D.oh, D.ow] += (
      cast(U, I[D.c, D.n, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW])
      * cast(U, K[D.c, D.f, D.kh, D.kw]))


@linalg_structured_op
def conv_2d_cnhw_chwf(
    I=TensorDef(TV.T1, S.C, S.N, S.OH * S.SH + S.KH * S.DH,
                S.OW * S.SW + S.KW * S.DW),
    K=TensorDef(TV.T2, S.C, S.KH, S.KW, S.F),
    O=TensorDef(U, S.N, S.OH, S.OW, S.F, output=True),
    strides=AttributeDef(S.SH, S.SW),
    dilations=AttributeDef(S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.oh, D.ow, D.f, D.c, D.kh, D.kw)
  O[D.n, D.oh, D.ow, D.f] += (
      cast(U, I[D.c, D.n, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW])
      * cast(U, K[D.c, D.kh, D.kw, D.f]))


@linalg_structured_op
def conv_2d_cnhw_fchw(
    I=TensorDef(TV.T1, S.C, S.N, S.OH * S.SH + S.KH * S.DH,
                S.OW * S.SW + S.KW * S.DW),
    K=TensorDef(TV.T2, S.F, S.C, S.KH, S.KW),
    O=TensorDef(U, S.N, S.F, S.OH, S.OW, output=True),
    strides=AttributeDef(S.SH, S.SW),
    dilations=AttributeDef(S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.f, D.oh, D.ow, D.c, D.kh, D.kw)
  O[D.n, D.f, D.oh, D.ow] += (
      cast(U, I[D.c, D.n, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW])
      * cast(U, K[D.f, D.c, D.kh, D.kw]))


@linalg_structured_op
def conv_2d_cnhw_fhwc(
    I=TensorDef(TV.T1, S.C, S.N, S.OH * S.SH + S.KH * S.DH,
                S.OW * S.SW + S.KW * S.DW),
    K=TensorDef(TV.T2, S.F, S.KH, S.KW, S.C),
    O=TensorDef(U, S.N, S.F, S.OH, S.OW, output=True),
    strides=AttributeDef(S.SH, S.SW),
    dilations=AttributeDef(S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.f, D.oh, D.ow, D.kh, D.kw, D.c)
  O[D.n, D.f, D.oh, D.ow] += (
      cast(U, I[D.c, D.n, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW])
      * cast(U, K[D.f, D.kh, D.kw, D.c]))


@linalg_structured_op
def conv_2d_cnhw_hwcf(
    I=TensorDef(TV.T1, S.C, S.N, S.OH * S.SH + S.KH * S.DH,
                S.OW * S.SW + S.KW * S.DW),
    K=TensorDef(TV.T2, S.KH, S.KW, S.C, S.F),
    O=TensorDef(U, S.N, S.OH, S.OW, S.F, output=True),
    strides=AttributeDef(S.SH, S.SW),
    dilations=AttributeDef(S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.oh, D.ow, D.f, D.kh, D.kw, D.c)
  O[D.n, D.oh, D.ow, D.f] += (
      cast(U, I[D.c, D.n, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW])
      * cast(U, K[D.kh, D.kw, D.c, D.f]))


@linalg_structured_op
def conv_2d_cnhw_hwfc(
    I=TensorDef(TV.T1, S.C, S.N, S.OH * S.SH + S.KH * S.DH,
                S.OW * S.SW + S.KW * S.DW),
    K=TensorDef(TV.T2, S.KH, S.KW, S.F, S.C),
    O=TensorDef(U, S.N, S.OH, S.OW, S.F, output=True),
    strides=AttributeDef(S.SH, S.SW),
    dilations=AttributeDef(S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.oh, D.ow, D.f, D.kh, D.kw, D.c)
  O[D.n, D.oh, D.ow, D.f] += (
      cast(U, I[D.c, D.n, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW])
      * cast(U, K[D.kh, D.kw, D.f, D.c]))


@linalg_structured_op
def conv_2d_chwn_cfhw(
    I=TensorDef(TV.T1, S.C, S.OH * S.SH + S.KH * S.DH,
                S.OW * S.SW + S.KW * S.DW, S.N),
    K=TensorDef(TV.T2, S.C, S.F, S.KH, S.KW),
    O=TensorDef(U, S.F, S.OH, S.OW, S.N, output=True),
    strides=AttributeDef(S.SH, S.SW),
    dilations=AttributeDef(S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.f, D.oh, D.ow, D.n, D.c, D.kh, D.kw)
  O[D.f, D.oh, D.ow, D.n] += (
      cast(U, I[D.c, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW, D.n])
      * cast(U, K[D.c, D.f, D.kh, D.kw]))


@linalg_structured_op
def conv_2d_chwn_chwf(
    I=TensorDef(TV.T1, S.C, S.OH * S.SH + S.KH * S.DH,
                S.OW * S.SW + S.KW * S.DW, S.N),
    K=TensorDef(TV.T2, S.C, S.KH, S.KW, S.F),
    O=TensorDef(U, S.OH, S.OW, S.N, S.F, output=True),
    strides=AttributeDef(S.SH, S.SW),
    dilations=AttributeDef(S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.oh, D.ow, D.n, D.f, D.c, D.kh, D.kw)
  O[D.oh, D.ow, D.n, D.f] += (
      cast(U, I[D.c, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW, D.n])
      * cast(U, K[D.c, D.kh, D.kw, D.f]))


@linalg_structured_op
def conv_2d_chwn_fchw(
    I=TensorDef(TV.T1, S.C, S.OH * S.SH + S.KH * S.DH,
                S.OW * S.SW + S.KW * S.DW, S.N),
    K=TensorDef(TV.T2, S.F, S.C, S.KH, S.KW),
    O=TensorDef(U, S.F, S.OH, S.OW, S.N, output=True),
    strides=AttributeDef(S.SH, S.SW),
    dilations=AttributeDef(S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.f, D.oh, D.ow, D.n, D.c, D.kh, D.kw)
  O[D.f, D.oh, D.ow, D.n] += (
      cast(U, I[D.c, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW, D.n])
      * cast(U, K[D.f, D.c, D.kh, D.kw]))


@linalg_structured_op
def conv_2d_chwn_fhwc(
    I=TensorDef(TV.T1, S.C, S.OH * S.SH + S.KH * S.DH,
                S.OW * S.SW + S.KW * S.DW, S.N),
    K=TensorDef(TV.T2, S.F, S.KH, S.KW, S.C),
    O=TensorDef(U, S.F, S.OH, S.OW, S.N, output=True),
    strides=AttributeDef(S.SH, S.SW),
    dilations=AttributeDef(S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.f, D.oh, D.ow, D.n, D.kh, D.kw, D.c)
  O[D.f, D.oh, D.ow, D.n] += (
      cast(U, I[D.c, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW, D.n])
      * cast(U, K[D.f, D.kh, D.kw, D.c]))


@linalg_structured_op
def conv_2d_chwn_hwcf(
    I=TensorDef(TV.T1, S.C, S.OH * S.SH + S.KH * S.DH,
                S.OW * S.SW + S.KW * S.DW, S.N),
    K=TensorDef(TV.T2, S.KH, S.KW, S.C, S.F),
    O=TensorDef(U, S.OH, S.OW, S.N, S.F, output=True),
    strides=AttributeDef(S.SH, S.SW),
    dilations=AttributeDef(S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.oh, D.ow, D.n, D.f, D.kh, D.kw, D.c)
  O[D.oh, D.ow, D.n, D.f] += (
      cast(U, I[D.c, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW, D.n])
      * cast(U, K[D.kh, D.kw, D.c, D.f]))


@linalg_structured_op
def conv_2d_chwn_hwfc(
    I=TensorDef(TV.T1, S.C, S.OH * S.SH + S.KH * S.DH,
                S.OW * S.SW + S.KW * S.DW, S.N),
    K=TensorDef(TV.T2, S.KH, S.KW, S.F, S.C),
    O=TensorDef(U, S.OH, S.OW, S.N, S.F, output=True),
    strides=AttributeDef(S.SH, S.SW),
    dilations=AttributeDef(S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.oh, D.ow, D.n, D.f, D.kh, D.kw, D.c)
  O[D.oh, D.ow, D.n, D.f] += (
      cast(U, I[D.c, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW, D.n])
      * cast(U, K[D.kh, D.kw, D.f, D.c]))


@linalg_structured_op
def conv_2d_hwnc_cfhw(
    I=TensorDef(TV.T1, S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW,
                S.N, S.C),
    K=TensorDef(TV.T2, S.C, S.F, S.KH, S.KW),
    O=TensorDef(U, S.F, S.OH, S.OW, S.N, output=True),
    strides=AttributeDef(S.SH, S.SW),
    dilations=AttributeDef(S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.f, D.oh, D.ow, D.n, D.c, D.kh, D.kw)
  O[D.f, D.oh, D.ow, D.n] += (
      cast(U, I[D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW, D.n, D.c])
      * cast(U, K[D.c, D.f, D.kh, D.kw]))


@linalg_structured_op
def conv_2d_hwnc_chwf(
    I=TensorDef(TV.T1, S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW,
                S.N, S.C),
    K=TensorDef(TV.T2, S.C, S.KH, S.KW, S.F),
    O=TensorDef(U, S.OH, S.OW, S.N, S.F, output=True),
    strides=AttributeDef(S.SH, S.SW),
    dilations=AttributeDef(S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.oh, D.ow, D.n, D.f, D.c, D.kh, D.kw)
  O[D.oh, D.ow, D.n, D.f] += (
      cast(U, I[D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW, D.n, D.c])
      * cast(U, K[D.c, D.kh, D.kw, D.f]))


@linalg_structured_op
def conv_2d_hwnc_fchw(
    I=TensorDef(TV.T1, S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW,
                S.N, S.C),
    K=TensorDef(TV.T2, S.F, S.C, S.KH, S.KW),
    O=TensorDef(U, S.F, S.OH, S.OW, S.N, output=True),
    strides=AttributeDef(S.SH, S.SW),
    dilations=AttributeDef(S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.f, D.oh, D.ow, D.n, D.c, D.kh, D.kw)
  O[D.f, D.oh, D.ow, D.n] += (
      cast(U, I[D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW, D.n, D.c])
      * cast(U, K[D.f, D.c, D.kh, D.kw]))


@linalg_structured_op
def conv_2d_hwnc_fhwc(
    I=TensorDef(TV.T1, S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW,
                S.N, S.C),
    K=TensorDef(TV.T2, S.F, S.KH, S.KW, S.C),
    O=TensorDef(U, S.F, S.OH, S.OW, S.N, output=True),
    strides=AttributeDef(S.SH, S.SW),
    dilations=AttributeDef(S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.f, D.oh, D.ow, D.n, D.kh, D.kw, D.c)
  O[D.f, D.oh, D.ow, D.n] += (
      cast(U, I[D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW, D.n, D.c])
      * cast(U, K[D.f, D.kh, D.kw, D.c]))


@linalg_structured_op
def conv_2d_hwnc_hwcf(
    I=TensorDef(TV.T1, S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW,
                S.N, S.C),
    K=TensorDef(TV.T2, S.KH, S.KW, S.C, S.F),
    O=TensorDef(U, S.OH, S.OW, S.N, S.F, output=True),
    strides=AttributeDef(S.SH, S.SW),
    dilations=AttributeDef(S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.oh, D.ow, D.n, D.f, D.kh, D.kw, D.c)
  O[D.oh, D.ow, D.n, D.f] += (
      cast(U, I[D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW, D.n, D.c])
      * cast(U, K[D.kh, D.kw, D.c, D.f]))


@linalg_structured_op
def conv_2d_hwnc_hwfc(
    I=TensorDef(TV.T1, S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW,
                S.N, S.C),
    K=TensorDef(TV.T2, S.KH, S.KW, S.F, S.C),
    O=TensorDef(U, S.OH, S.OW, S.N, S.F, output=True),
    strides=AttributeDef(S.SH, S.SW),
    dilations=AttributeDef(S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.oh, D.ow, D.n, D.f, D.kh, D.kw, D.c)
  O[D.oh, D.ow, D.n, D.f] += (
      cast(U, I[D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW, D.n, D.c])
      * cast(U, K[D.kh, D.kw, D.f, D.c]))


@linalg_structured_op
def conv_2d_hwcn_cfhw(
    I=TensorDef(TV.T1, S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW,
                S.C, S.N),
    K=TensorDef(TV.T2, S.C, S.F, S.KH, S.KW),
    O=TensorDef(U, S.F, S.OH, S.OW, S.N, output=True),
    strides=AttributeDef(S.SH, S.SW),
    dilations=AttributeDef(S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.f, D.oh, D.ow, D.n, D.c, D.kh, D.kw)
  O[D.f, D.oh, D.ow, D.n] += (
      cast(U, I[D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW, D.c, D.n])
      * cast(U, K[D.c, D.f, D.kh, D.kw]))


@linalg_structured_op
def conv_2d_hwcn_chwf(
    I=TensorDef(TV.T1, S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW,
                S.C, S.N),
    K=TensorDef(TV.T2, S.C, S.KH, S.KW, S.F),
    O=TensorDef(U, S.OH, S.OW, S.N, S.F, output=True),
    strides=AttributeDef(S.SH, S.SW),
    dilations=AttributeDef(S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.oh, D.ow, D.n, D.f, D.c, D.kh, D.kw)
  O[D.oh, D.ow, D.n, D.f] += (
      cast(U, I[D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW, D.c, D.n])
      * cast(U, K[D.c, D.kh, D.kw, D.f]))


@linalg_structured_op
def conv_2d_hwcn_fchw(
    I=TensorDef(TV.T1, S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW,
                S.C, S.N),
    K=TensorDef(TV.T2, S.F, S.C, S.KH, S.KW),
    O=TensorDef(U, S.F, S.OH, S.OW, S.N, output=True),
    strides=AttributeDef(S.SH, S.SW),
    dilations=AttributeDef(S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.f, D.oh, D.ow, D.n, D.c, D.kh, D.kw)
  O[D.f, D.oh, D.ow, D.n] += (
      cast(U, I[D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW, D.c, D.n])
      * cast(U, K[D.f, D.c, D.kh, D.kw]))


@linalg_structured_op
def conv_2d_hwcn_fhwc(
    I=TensorDef(TV.T1, S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW,
                S.C, S.N),
    K=TensorDef(TV.T2, S.F, S.KH, S.KW, S.C),
    O=TensorDef(U, S.F, S.OH, S.OW, S.N, output=True),
    strides=AttributeDef(S.SH, S.SW),
    dilations=AttributeDef(S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.f, D.oh, D.ow, D.n, D.kh, D.kw, D.c)
  O[D.f, D.oh, D.ow, D.n] += (
      cast(U, I[D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW, D.c, D.n])
      * cast(U, K[D.f, D.kh, D.kw, D.c]))


@linalg_structured_op
def conv_2d_hwcn_hwcf(
    I=TensorDef(TV.T1, S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW,
                S.C, S.N),
    K=TensorDef(TV.T2, S.KH, S.KW, S.C, S.F),
    O=TensorDef(U, S.OH, S.OW, S.N, S.F, output=True),
    strides=AttributeDef(S.SH, S.SW),
    dilations=AttributeDef(S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.oh, D.ow, D.n, D.f, D.kh, D.kw, D.c)
  O[D.oh, D.ow, D.n, D.f] += (
      cast(U, I[D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW, D.c, D.n])
      * cast(U, K[D.kh, D.kw, D.c, D.f]))


@linalg_structured_op
def conv_2d_hwcn_hwfc(
    I=TensorDef(TV.T1, S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW,
                S.C, S.N),
    K=TensorDef(TV.T2, S.KH, S.KW, S.F, S.C),
    O=TensorDef(U, S.OH, S.OW, S.N, S.F, output=True),
    strides=AttributeDef(S.SH, S.SW),
    dilations=AttributeDef(S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.oh, D.ow, D.n, D.f, D.kh, D.kw, D.c)
  O[D.oh, D.ow, D.n, D.f] += (
      cast(U, I[D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW, D.c, D.n])
      * cast(U, K[D.kh, D.kw, D.f, D.c]))


@linalg_structured_op
def conv_3d_ncdhw_cfdhw(
    I=TensorDef(TV.T1, S.N, S.C, S.OD * S.SD + S.KD * S.DD,
                S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW),
    K=TensorDef(TV.T2, S.C, S.F, S.KD, S.KH, S.KW),
    O=TensorDef(U, S.N, S.F, S.OD, S.OH, S.OW, output=True),
    strides=AttributeDef(S.SD, S.SH, S.SW),
    dilations=AttributeDef(S.DD, S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.f, D.od, D.oh, D.ow, D.c, D.kd, D.kh, D.kw)
  O[D.n, D.f, D.od, D.oh, D.ow] += (
      cast(
          U, I[D.n, D.c, D.od * S.SD + D.kd * S.DD, D.oh * S.SH + D.kh * S.DH,
               D.ow * S.SW + D.kw * S.DW]) *
      cast(U, K[D.c, D.f, D.kd, D.kh, D.kw]))


@linalg_structured_op
def conv_3d_ncdhw_cdhwf(
    I=TensorDef(TV.T1, S.N, S.C, S.OD * S.SD + S.KD * S.DD,
                S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW),
    K=TensorDef(TV.T2, S.C, S.KD, S.KH, S.KW, S.F),
    O=TensorDef(U, S.N, S.OD, S.OH, S.OW, S.F, output=True),
    strides=AttributeDef(S.SD, S.SH, S.SW),
    dilations=AttributeDef(S.DD, S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.od, D.oh, D.ow, D.f, D.c, D.kd, D.kh, D.kw)
  O[D.n, D.od, D.oh, D.ow, D.f] += (
      cast(
          U, I[D.n, D.c, D.od * S.SD + D.kd * S.DD, D.oh * S.SH + D.kh * S.DH,
               D.ow * S.SW + D.kw * S.DW]) *
      cast(U, K[D.c, D.kd, D.kh, D.kw, D.f]))


@linalg_structured_op
def conv_3d_ncdhw_fcdhw(
    I=TensorDef(TV.T1, S.N, S.C, S.OD * S.SD + S.KD * S.DD,
                S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW),
    K=TensorDef(TV.T2, S.F, S.C, S.KD, S.KH, S.KW),
    O=TensorDef(U, S.N, S.F, S.OD, S.OH, S.OW, output=True),
    strides=AttributeDef(S.SD, S.SH, S.SW),
    dilations=AttributeDef(S.DD, S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.f, D.od, D.oh, D.ow, D.c, D.kd, D.kh, D.kw)
  O[D.n, D.f, D.od, D.oh, D.ow] += (
      cast(
          U, I[D.n, D.c, D.od * S.SD + D.kd * S.DD, D.oh * S.SH + D.kh * S.DH,
               D.ow * S.SW + D.kw * S.DW]) *
      cast(U, K[D.f, D.c, D.kd, D.kh, D.kw]))


@linalg_structured_op
def conv_3d_ncdhw_fdhwc(
    I=TensorDef(TV.T1, S.N, S.C, S.OD * S.SD + S.KD * S.DD,
                S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW),
    K=TensorDef(TV.T2, S.F, S.KD, S.KH, S.KW, S.C),
    O=TensorDef(U, S.N, S.F, S.OD, S.OH, S.OW, output=True),
    strides=AttributeDef(S.SD, S.SH, S.SW),
    dilations=AttributeDef(S.DD, S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.f, D.od, D.oh, D.ow, D.kd, D.kh, D.kw, D.c)
  O[D.n, D.f, D.od, D.oh, D.ow] += (
      cast(
          U, I[D.n, D.c, D.od * S.SD + D.kd * S.DD, D.oh * S.SH + D.kh * S.DH,
               D.ow * S.SW + D.kw * S.DW]) *
      cast(U, K[D.f, D.kd, D.kh, D.kw, D.c]))


@linalg_structured_op
def conv_3d_ncdhw_dhwcf(
    I=TensorDef(TV.T1, S.N, S.C, S.OD * S.SD + S.KD * S.DD,
                S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW),
    K=TensorDef(TV.T2, S.KD, S.KH, S.KW, S.C, S.F),
    O=TensorDef(U, S.N, S.OD, S.OH, S.OW, S.F, output=True),
    strides=AttributeDef(S.SD, S.SH, S.SW),
    dilations=AttributeDef(S.DD, S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.od, D.oh, D.ow, D.f, D.kd, D.kh, D.kw, D.c)
  O[D.n, D.od, D.oh, D.ow, D.f] += (
      cast(
          U, I[D.n, D.c, D.od * S.SD + D.kd * S.DD, D.oh * S.SH + D.kh * S.DH,
               D.ow * S.SW + D.kw * S.DW]) *
      cast(U, K[D.kd, D.kh, D.kw, D.c, D.f]))


@linalg_structured_op
def conv_3d_ncdhw_dhwfc(
    I=TensorDef(TV.T1, S.N, S.C, S.OD * S.SD + S.KD * S.DD,
                S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW),
    K=TensorDef(TV.T2, S.KD, S.KH, S.KW, S.F, S.C),
    O=TensorDef(U, S.N, S.OD, S.OH, S.OW, S.F, output=True),
    strides=AttributeDef(S.SD, S.SH, S.SW),
    dilations=AttributeDef(S.DD, S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.od, D.oh, D.ow, D.f, D.kd, D.kh, D.kw, D.c)
  O[D.n, D.od, D.oh, D.ow, D.f] += (
      cast(
          U, I[D.n, D.c, D.od * S.SD + D.kd * S.DD, D.oh * S.SH + D.kh * S.DH,
               D.ow * S.SW + D.kw * S.DW]) *
      cast(U, K[D.kd, D.kh, D.kw, D.f, D.c]))


@linalg_structured_op
def conv_3d_ndhwc_cfdhw(
    I=TensorDef(TV.T1, S.N, S.OD * S.SD + S.KD * S.DD,
                S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW, S.C),
    K=TensorDef(TV.T2, S.C, S.F, S.KD, S.KH, S.KW),
    O=TensorDef(U, S.N, S.F, S.OD, S.OH, S.OW, output=True),
    strides=AttributeDef(S.SD, S.SH, S.SW),
    dilations=AttributeDef(S.DD, S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.f, D.od, D.oh, D.ow, D.c, D.kd, D.kh, D.kw)
  O[D.n, D.f, D.od, D.oh, D.ow] += (
      cast(
          U, I[D.n, D.od * S.SD + D.kd * S.DD, D.oh * S.SH + D.kh * S.DH,
               D.ow * S.SW + D.kw * S.DW, D.c]) *
      cast(U, K[D.c, D.f, D.kd, D.kh, D.kw]))


@linalg_structured_op
def conv_3d_ndhwc_cdhwf(
    I=TensorDef(TV.T1, S.N, S.OD * S.SD + S.KD * S.DD,
                S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW, S.C),
    K=TensorDef(TV.T2, S.C, S.KD, S.KH, S.KW, S.F),
    O=TensorDef(U, S.N, S.OD, S.OH, S.OW, S.F, output=True),
    strides=AttributeDef(S.SD, S.SH, S.SW),
    dilations=AttributeDef(S.DD, S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.od, D.oh, D.ow, D.f, D.c, D.kd, D.kh, D.kw)
  O[D.n, D.od, D.oh, D.ow, D.f] += (
      cast(
          U, I[D.n, D.od * S.SD + D.kd * S.DD, D.oh * S.SH + D.kh * S.DH,
               D.ow * S.SW + D.kw * S.DW, D.c]) *
      cast(U, K[D.c, D.kd, D.kh, D.kw, D.f]))


@linalg_structured_op
def conv_3d_ndhwc_fcdhw(
    I=TensorDef(TV.T1, S.N, S.OD * S.SD + S.KD * S.DD,
                S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW, S.C),
    K=TensorDef(TV.T2, S.F, S.C, S.KD, S.KH, S.KW),
    O=TensorDef(U, S.N, S.F, S.OD, S.OH, S.OW, output=True),
    strides=AttributeDef(S.SD, S.SH, S.SW),
    dilations=AttributeDef(S.DD, S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.f, D.od, D.oh, D.ow, D.c, D.kd, D.kh, D.kw)
  O[D.n, D.f, D.od, D.oh, D.ow] += (
      cast(
          U, I[D.n, D.od * S.SD + D.kd * S.DD, D.oh * S.SH + D.kh * S.DH,
               D.ow * S.SW + D.kw * S.DW, D.c]) *
      cast(U, K[D.f, D.c, D.kd, D.kh, D.kw]))


@linalg_structured_op
def conv_3d_ndhwc_fdhwc(
    I=TensorDef(TV.T1, S.N, S.OD * S.SD + S.KD * S.DD,
                S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW, S.C),
    K=TensorDef(TV.T2, S.F, S.KD, S.KH, S.KW, S.C),
    O=TensorDef(U, S.N, S.F, S.OD, S.OH, S.OW, output=True),
    strides=AttributeDef(S.SD, S.SH, S.SW),
    dilations=AttributeDef(S.DD, S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.f, D.od, D.oh, D.ow, D.kd, D.kh, D.kw, D.c)
  O[D.n, D.f, D.od, D.oh, D.ow] += (
      cast(
          U, I[D.n, D.od * S.SD + D.kd * S.DD, D.oh * S.SH + D.kh * S.DH,
               D.ow * S.SW + D.kw * S.DW, D.c]) *
      cast(U, K[D.f, D.kd, D.kh, D.kw, D.c]))


@linalg_structured_op
def conv_3d_ndhwc_dhwcf(
    I=TensorDef(TV.T1, S.N, S.OD * S.SD + S.KD * S.DD,
                S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW, S.C),
    K=TensorDef(TV.T2, S.KD, S.KH, S.KW, S.C, S.F),
    O=TensorDef(U, S.N, S.OD, S.OH, S.OW, S.F, output=True),
    strides=AttributeDef(S.SD, S.SH, S.SW),
    dilations=AttributeDef(S.DD, S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.od, D.oh, D.ow, D.f, D.kd, D.kh, D.kw, D.c)
  O[D.n, D.od, D.oh, D.ow, D.f] += (
      cast(
          U, I[D.n, D.od * S.SD + D.kd * S.DD, D.oh * S.SH + D.kh * S.DH,
               D.ow * S.SW + D.kw * S.DW, D.c]) *
      cast(U, K[D.kd, D.kh, D.kw, D.c, D.f]))


@linalg_structured_op
def conv_3d_ndhwc_dhwfc(
    I=TensorDef(TV.T1, S.N, S.OD * S.SD + S.KD * S.DD,
                S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW, S.C),
    K=TensorDef(TV.T2, S.KD, S.KH, S.KW, S.F, S.C),
    O=TensorDef(U, S.N, S.OD, S.OH, S.OW, S.F, output=True),
    strides=AttributeDef(S.SD, S.SH, S.SW),
    dilations=AttributeDef(S.DD, S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.od, D.oh, D.ow, D.f, D.kd, D.kh, D.kw, D.c)
  O[D.n, D.od, D.oh, D.ow, D.f] += (
      cast(
          U, I[D.n, D.od * S.SD + D.kd * S.DD, D.oh * S.SH + D.kh * S.DH,
               D.ow * S.SW + D.kw * S.DW, D.c]) *
      cast(U, K[D.kd, D.kh, D.kw, D.f, D.c]))


@linalg_structured_op
def conv_3d_cndhw_cfdhw(
    I=TensorDef(TV.T1, S.C, S.N, S.OD * S.SD + S.KD * S.DD,
                S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW),
    K=TensorDef(TV.T2, S.C, S.F, S.KD, S.KH, S.KW),
    O=TensorDef(U, S.N, S.F, S.OD, S.OH, S.OW, output=True),
    strides=AttributeDef(S.SD, S.SH, S.SW),
    dilations=AttributeDef(S.DD, S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.f, D.od, D.oh, D.ow, D.c, D.kd, D.kh, D.kw)
  O[D.n, D.f, D.od, D.oh, D.ow] += (
      cast(
          U, I[D.c, D.n, D.od * S.SD + D.kd * S.DD, D.oh * S.SH + D.kh * S.DH,
               D.ow * S.SW + D.kw * S.DW]) *
      cast(U, K[D.c, D.f, D.kd, D.kh, D.kw]))


@linalg_structured_op
def conv_3d_cndhw_cdhwf(
    I=TensorDef(TV.T1, S.C, S.N, S.OD * S.SD + S.KD * S.DD,
                S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW),
    K=TensorDef(TV.T2, S.C, S.KD, S.KH, S.KW, S.F),
    O=TensorDef(U, S.N, S.OD, S.OH, S.OW, S.F, output=True),
    strides=AttributeDef(S.SD, S.SH, S.SW),
    dilations=AttributeDef(S.DD, S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.od, D.oh, D.ow, D.f, D.c, D.kd, D.kh, D.kw)
  O[D.n, D.od, D.oh, D.ow, D.f] += (
      cast(
          U, I[D.c, D.n, D.od * S.SD + D.kd * S.DD, D.oh * S.SH + D.kh * S.DH,
               D.ow * S.SW + D.kw * S.DW]) *
      cast(U, K[D.c, D.kd, D.kh, D.kw, D.f]))


@linalg_structured_op
def conv_3d_cndhw_fcdhw(
    I=TensorDef(TV.T1, S.C, S.N, S.OD * S.SD + S.KD * S.DD,
                S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW),
    K=TensorDef(TV.T2, S.F, S.C, S.KD, S.KH, S.KW),
    O=TensorDef(U, S.N, S.F, S.OD, S.OH, S.OW, output=True),
    strides=AttributeDef(S.SD, S.SH, S.SW),
    dilations=AttributeDef(S.DD, S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.f, D.od, D.oh, D.ow, D.c, D.kd, D.kh, D.kw)
  O[D.n, D.f, D.od, D.oh, D.ow] += (
      cast(
          U, I[D.c, D.n, D.od * S.SD + D.kd * S.DD, D.oh * S.SH + D.kh * S.DH,
               D.ow * S.SW + D.kw * S.DW]) *
      cast(U, K[D.f, D.c, D.kd, D.kh, D.kw]))


@linalg_structured_op
def conv_3d_cndhw_fdhwc(
    I=TensorDef(TV.T1, S.C, S.N, S.OD * S.SD + S.KD * S.DD,
                S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW),
    K=TensorDef(TV.T2, S.F, S.KD, S.KH, S.KW, S.C),
    O=TensorDef(U, S.N, S.F, S.OD, S.OH, S.OW, output=True),
    strides=AttributeDef(S.SD, S.SH, S.SW),
    dilations=AttributeDef(S.DD, S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.f, D.od, D.oh, D.ow, D.kd, D.kh, D.kw, D.c)
  O[D.n, D.f, D.od, D.oh, D.ow] += (
      cast(
          U, I[D.c, D.n, D.od * S.SD + D.kd * S.DD, D.oh * S.SH + D.kh * S.DH,
               D.ow * S.SW + D.kw * S.DW]) *
      cast(U, K[D.f, D.kd, D.kh, D.kw, D.c]))


@linalg_structured_op
def conv_3d_cndhw_dhwcf(
    I=TensorDef(TV.T1, S.C, S.N, S.OD * S.SD + S.KD * S.DD,
                S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW),
    K=TensorDef(TV.T2, S.KD, S.KH, S.KW, S.C, S.F),
    O=TensorDef(U, S.N, S.OD, S.OH, S.OW, S.F, output=True),
    strides=AttributeDef(S.SD, S.SH, S.SW),
    dilations=AttributeDef(S.DD, S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.od, D.oh, D.ow, D.f, D.kd, D.kh, D.kw, D.c)
  O[D.n, D.od, D.oh, D.ow, D.f] += (
      cast(
          U, I[D.c, D.n, D.od * S.SD + D.kd * S.DD, D.oh * S.SH + D.kh * S.DH,
               D.ow * S.SW + D.kw * S.DW]) *
      cast(U, K[D.kd, D.kh, D.kw, D.c, D.f]))


@linalg_structured_op
def conv_3d_cndhw_dhwfc(
    I=TensorDef(TV.T1, S.C, S.N, S.OD * S.SD + S.KD * S.DD,
                S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW),
    K=TensorDef(TV.T2, S.KD, S.KH, S.KW, S.F, S.C),
    O=TensorDef(U, S.N, S.OD, S.OH, S.OW, S.F, output=True),
    strides=AttributeDef(S.SD, S.SH, S.SW),
    dilations=AttributeDef(S.DD, S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.n, D.od, D.oh, D.ow, D.f, D.kd, D.kh, D.kw, D.c)
  O[D.n, D.od, D.oh, D.ow, D.f] += (
      cast(
          U, I[D.c, D.n, D.od * S.SD + D.kd * S.DD, D.oh * S.SH + D.kh * S.DH,
               D.ow * S.SW + D.kw * S.DW]) *
      cast(U, K[D.kd, D.kh, D.kw, D.f, D.c]))


@linalg_structured_op
def conv_3d_cdhwn_cfdhw(
    I=TensorDef(TV.T1, S.C, S.OD * S.SD + S.KD * S.DD,
                S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW, S.N),
    K=TensorDef(TV.T2, S.C, S.F, S.KD, S.KH, S.KW),
    O=TensorDef(U, S.F, S.OD, S.OH, S.OW, S.N, output=True),
    strides=AttributeDef(S.SD, S.SH, S.SW),
    dilations=AttributeDef(S.DD, S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.f, D.od, D.oh, D.ow, D.n, D.c, D.kd, D.kh, D.kw)
  O[D.f, D.od, D.oh, D.ow, D.n] += (
      cast(
          U, I[D.c, D.od * S.SD + D.kd * S.DD, D.oh * S.SH + D.kh * S.DH,
               D.ow * S.SW + D.kw * S.DW, D.n]) *
      cast(U, K[D.c, D.f, D.kd, D.kh, D.kw]))


@linalg_structured_op
def conv_3d_cdhwn_cdhwf(
    I=TensorDef(TV.T1, S.C, S.OD * S.SD + S.KD * S.DD,
                S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW, S.N),
    K=TensorDef(TV.T2, S.C, S.KD, S.KH, S.KW, S.F),
    O=TensorDef(U, S.OD, S.OH, S.OW, S.N, S.F, output=True),
    strides=AttributeDef(S.SD, S.SH, S.SW),
    dilations=AttributeDef(S.DD, S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.od, D.oh, D.ow, D.n, D.f, D.c, D.kd, D.kh, D.kw)
  O[D.od, D.oh, D.ow, D.n, D.f] += (
      cast(
          U, I[D.c, D.od * S.SD + D.kd * S.DD, D.oh * S.SH + D.kh * S.DH,
               D.ow * S.SW + D.kw * S.DW, D.n]) *
      cast(U, K[D.c, D.kd, D.kh, D.kw, D.f]))


@linalg_structured_op
def conv_3d_cdhwn_fcdhw(
    I=TensorDef(TV.T1, S.C, S.OD * S.SD + S.KD * S.DD,
                S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW, S.N),
    K=TensorDef(TV.T2, S.F, S.C, S.KD, S.KH, S.KW),
    O=TensorDef(U, S.F, S.OD, S.OH, S.OW, S.N, output=True),
    strides=AttributeDef(S.SD, S.SH, S.SW),
    dilations=AttributeDef(S.DD, S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.f, D.od, D.oh, D.ow, D.n, D.c, D.kd, D.kh, D.kw)
  O[D.f, D.od, D.oh, D.ow, D.n] += (
      cast(
          U, I[D.c, D.od * S.SD + D.kd * S.DD, D.oh * S.SH + D.kh * S.DH,
               D.ow * S.SW + D.kw * S.DW, D.n]) *
      cast(U, K[D.f, D.c, D.kd, D.kh, D.kw]))


@linalg_structured_op
def conv_3d_cdhwn_fdhwc(
    I=TensorDef(TV.T1, S.C, S.OD * S.SD + S.KD * S.DD,
                S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW, S.N),
    K=TensorDef(TV.T2, S.F, S.KD, S.KH, S.KW, S.C),
    O=TensorDef(U, S.F, S.OD, S.OH, S.OW, S.N, output=True),
    strides=AttributeDef(S.SD, S.SH, S.SW),
    dilations=AttributeDef(S.DD, S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.f, D.od, D.oh, D.ow, D.n, D.kd, D.kh, D.kw, D.c)
  O[D.f, D.od, D.oh, D.ow, D.n] += (
      cast(
          U, I[D.c, D.od * S.SD + D.kd * S.DD, D.oh * S.SH + D.kh * S.DH,
               D.ow * S.SW + D.kw * S.DW, D.n]) *
      cast(U, K[D.f, D.kd, D.kh, D.kw, D.c]))


@linalg_structured_op
def conv_3d_cdhwn_dhwcf(
    I=TensorDef(TV.T1, S.C, S.OD * S.SD + S.KD * S.DD,
                S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW, S.N),
    K=TensorDef(TV.T2, S.KD, S.KH, S.KW, S.C, S.F),
    O=TensorDef(U, S.OD, S.OH, S.OW, S.N, S.F, output=True),
    strides=AttributeDef(S.SD, S.SH, S.SW),
    dilations=AttributeDef(S.DD, S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.od, D.oh, D.ow, D.n, D.f, D.kd, D.kh, D.kw, D.c)
  O[D.od, D.oh, D.ow, D.n, D.f] += (
      cast(
          U, I[D.c, D.od * S.SD + D.kd * S.DD, D.oh * S.SH + D.kh * S.DH,
               D.ow * S.SW + D.kw * S.DW, D.n]) *
      cast(U, K[D.kd, D.kh, D.kw, D.c, D.f]))


@linalg_structured_op
def conv_3d_cdhwn_dhwfc(
    I=TensorDef(TV.T1, S.C, S.OD * S.SD + S.KD * S.DD,
                S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW, S.N),
    K=TensorDef(TV.T2, S.KD, S.KH, S.KW, S.F, S.C),
    O=TensorDef(U, S.OD, S.OH, S.OW, S.N, S.F, output=True),
    strides=AttributeDef(S.SD, S.SH, S.SW),
    dilations=AttributeDef(S.DD, S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.od, D.oh, D.ow, D.n, D.f, D.kd, D.kh, D.kw, D.c)
  O[D.od, D.oh, D.ow, D.n, D.f] += (
      cast(
          U, I[D.c, D.od * S.SD + D.kd * S.DD, D.oh * S.SH + D.kh * S.DH,
               D.ow * S.SW + D.kw * S.DW, D.n]) *
      cast(U, K[D.kd, D.kh, D.kw, D.f, D.c]))


@linalg_structured_op
def conv_3d_dhwnc_cfdhw(
    I=TensorDef(TV.T1, S.OD * S.SD + S.KD * S.DD, S.OH * S.SH + S.KH * S.DH,
                S.OW * S.SW + S.KW * S.DW, S.N, S.C),
    K=TensorDef(TV.T2, S.C, S.F, S.KD, S.KH, S.KW),
    O=TensorDef(U, S.F, S.OD, S.OH, S.OW, S.N, output=True),
    strides=AttributeDef(S.SD, S.SH, S.SW),
    dilations=AttributeDef(S.DD, S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.f, D.od, D.oh, D.ow, D.n, D.c, D.kd, D.kh, D.kw)
  O[D.f, D.od, D.oh, D.ow, D.n] += (
      cast(
          U, I[D.od * S.SD + D.kd * S.DD, D.oh * S.SH + D.kh * S.DH,
               D.ow * S.SW + D.kw * S.DW, D.n, D.c]) *
      cast(U, K[D.c, D.f, D.kd, D.kh, D.kw]))


@linalg_structured_op
def conv_3d_dhwnc_cdhwf(
    I=TensorDef(TV.T1, S.OD * S.SD + S.KD * S.DD, S.OH * S.SH + S.KH * S.DH,
                S.OW * S.SW + S.KW * S.DW, S.N, S.C),
    K=TensorDef(TV.T2, S.C, S.KD, S.KH, S.KW, S.F),
    O=TensorDef(U, S.OD, S.OH, S.OW, S.N, S.F, output=True),
    strides=AttributeDef(S.SD, S.SH, S.SW),
    dilations=AttributeDef(S.DD, S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.od, D.oh, D.ow, D.n, D.f, D.c, D.kd, D.kh, D.kw)
  O[D.od, D.oh, D.ow, D.n, D.f] += (
      cast(
          U, I[D.od * S.SD + D.kd * S.DD, D.oh * S.SH + D.kh * S.DH,
               D.ow * S.SW + D.kw * S.DW, D.n, D.c]) *
      cast(U, K[D.c, D.kd, D.kh, D.kw, D.f]))


@linalg_structured_op
def conv_3d_dhwnc_fcdhw(
    I=TensorDef(TV.T1, S.OD * S.SD + S.KD * S.DD, S.OH * S.SH + S.KH * S.DH,
                S.OW * S.SW + S.KW * S.DW, S.N, S.C),
    K=TensorDef(TV.T2, S.F, S.C, S.KD, S.KH, S.KW),
    O=TensorDef(U, S.F, S.OD, S.OH, S.OW, S.N, output=True),
    strides=AttributeDef(S.SD, S.SH, S.SW),
    dilations=AttributeDef(S.DD, S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.f, D.od, D.oh, D.ow, D.n, D.c, D.kd, D.kh, D.kw)
  O[D.f, D.od, D.oh, D.ow, D.n] += (
      cast(
          U, I[D.od * S.SD + D.kd * S.DD, D.oh * S.SH + D.kh * S.DH,
               D.ow * S.SW + D.kw * S.DW, D.n, D.c]) *
      cast(U, K[D.f, D.c, D.kd, D.kh, D.kw]))


@linalg_structured_op
def conv_3d_dhwnc_fdhwc(
    I=TensorDef(TV.T1, S.OD * S.SD + S.KD * S.DD, S.OH * S.SH + S.KH * S.DH,
                S.OW * S.SW + S.KW * S.DW, S.N, S.C),
    K=TensorDef(TV.T2, S.F, S.KD, S.KH, S.KW, S.C),
    O=TensorDef(U, S.F, S.OD, S.OH, S.OW, S.N, output=True),
    strides=AttributeDef(S.SD, S.SH, S.SW),
    dilations=AttributeDef(S.DD, S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.f, D.od, D.oh, D.ow, D.n, D.kd, D.kh, D.kw, D.c)
  O[D.f, D.od, D.oh, D.ow, D.n] += (
      cast(
          U, I[D.od * S.SD + D.kd * S.DD, D.oh * S.SH + D.kh * S.DH,
               D.ow * S.SW + D.kw * S.DW, D.n, D.c]) *
      cast(U, K[D.f, D.kd, D.kh, D.kw, D.c]))


@linalg_structured_op
def conv_3d_dhwnc_dhwcf(
    I=TensorDef(TV.T1, S.OD * S.SD + S.KD * S.DD, S.OH * S.SH + S.KH * S.DH,
                S.OW * S.SW + S.KW * S.DW, S.N, S.C),
    K=TensorDef(TV.T2, S.KD, S.KH, S.KW, S.C, S.F),
    O=TensorDef(U, S.OD, S.OH, S.OW, S.N, S.F, output=True),
    strides=AttributeDef(S.SD, S.SH, S.SW),
    dilations=AttributeDef(S.DD, S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.od, D.oh, D.ow, D.n, D.f, D.kd, D.kh, D.kw, D.c)
  O[D.od, D.oh, D.ow, D.n, D.f] += (
      cast(
          U, I[D.od * S.SD + D.kd * S.DD, D.oh * S.SH + D.kh * S.DH,
               D.ow * S.SW + D.kw * S.DW, D.n, D.c]) *
      cast(U, K[D.kd, D.kh, D.kw, D.c, D.f]))


@linalg_structured_op
def conv_3d_dhwnc_dhwfc(
    I=TensorDef(TV.T1, S.OD * S.SD + S.KD * S.DD, S.OH * S.SH + S.KH * S.DH,
                S.OW * S.SW + S.KW * S.DW, S.N, S.C),
    K=TensorDef(TV.T2, S.KD, S.KH, S.KW, S.F, S.C),
    O=TensorDef(U, S.OD, S.OH, S.OW, S.N, S.F, output=True),
    strides=AttributeDef(S.SD, S.SH, S.SW),
    dilations=AttributeDef(S.DD, S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.od, D.oh, D.ow, D.n, D.f, D.kd, D.kh, D.kw, D.c)
  O[D.od, D.oh, D.ow, D.n, D.f] += (
      cast(
          U, I[D.od * S.SD + D.kd * S.DD, D.oh * S.SH + D.kh * S.DH,
               D.ow * S.SW + D.kw * S.DW, D.n, D.c]) *
      cast(U, K[D.kd, D.kh, D.kw, D.f, D.c]))


@linalg_structured_op
def conv_3d_dhwcn_cfdhw(
    I=TensorDef(TV.T1, S.OD * S.SD + S.KD * S.DD, S.OH * S.SH + S.KH * S.DH,
                S.OW * S.SW + S.KW * S.DW, S.C, S.N),
    K=TensorDef(TV.T2, S.C, S.F, S.KD, S.KH, S.KW),
    O=TensorDef(U, S.F, S.OD, S.OH, S.OW, S.N, output=True),
    strides=AttributeDef(S.SD, S.SH, S.SW),
    dilations=AttributeDef(S.DD, S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.f, D.od, D.oh, D.ow, D.n, D.c, D.kd, D.kh, D.kw)
  O[D.f, D.od, D.oh, D.ow, D.n] += (
      cast(
          U, I[D.od * S.SD + D.kd * S.DD, D.oh * S.SH + D.kh * S.DH,
               D.ow * S.SW + D.kw * S.DW, D.c, D.n]) *
      cast(U, K[D.c, D.f, D.kd, D.kh, D.kw]))


@linalg_structured_op
def conv_3d_dhwcn_cdhwf(
    I=TensorDef(TV.T1, S.OD * S.SD + S.KD * S.DD, S.OH * S.SH + S.KH * S.DH,
                S.OW * S.SW + S.KW * S.DW, S.C, S.N),
    K=TensorDef(TV.T2, S.C, S.KD, S.KH, S.KW, S.F),
    O=TensorDef(U, S.OD, S.OH, S.OW, S.N, S.F, output=True),
    strides=AttributeDef(S.SD, S.SH, S.SW),
    dilations=AttributeDef(S.DD, S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.od, D.oh, D.ow, D.n, D.f, D.c, D.kd, D.kh, D.kw)
  O[D.od, D.oh, D.ow, D.n, D.f] += (
      cast(
          U, I[D.od * S.SD + D.kd * S.DD, D.oh * S.SH + D.kh * S.DH,
               D.ow * S.SW + D.kw * S.DW, D.c, D.n]) *
      cast(U, K[D.c, D.kd, D.kh, D.kw, D.f]))


@linalg_structured_op
def conv_3d_dhwcn_fcdhw(
    I=TensorDef(TV.T1, S.OD * S.SD + S.KD * S.DD, S.OH * S.SH + S.KH * S.DH,
                S.OW * S.SW + S.KW * S.DW, S.C, S.N),
    K=TensorDef(TV.T2, S.F, S.C, S.KD, S.KH, S.KW),
    O=TensorDef(U, S.F, S.OD, S.OH, S.OW, S.N, output=True),
    strides=AttributeDef(S.SD, S.SH, S.SW),
    dilations=AttributeDef(S.DD, S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.f, D.od, D.oh, D.ow, D.n, D.c, D.kd, D.kh, D.kw)
  O[D.f, D.od, D.oh, D.ow, D.n] += (
      cast(
          U, I[D.od * S.SD + D.kd * S.DD, D.oh * S.SH + D.kh * S.DH,
               D.ow * S.SW + D.kw * S.DW, D.c, D.n]) *
      cast(U, K[D.f, D.c, D.kd, D.kh, D.kw]))


@linalg_structured_op
def conv_3d_dhwcn_fdhwc(
    I=TensorDef(TV.T1, S.OD * S.SD + S.KD * S.DD, S.OH * S.SH + S.KH * S.DH,
                S.OW * S.SW + S.KW * S.DW, S.C, S.N),
    K=TensorDef(TV.T2, S.F, S.KD, S.KH, S.KW, S.C),
    O=TensorDef(U, S.F, S.OD, S.OH, S.OW, S.N, output=True),
    strides=AttributeDef(S.SD, S.SH, S.SW),
    dilations=AttributeDef(S.DD, S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.f, D.od, D.oh, D.ow, D.n, D.kd, D.kh, D.kw, D.c)
  O[D.f, D.od, D.oh, D.ow, D.n] += (
      cast(
          U, I[D.od * S.SD + D.kd * S.DD, D.oh * S.SH + D.kh * S.DH,
               D.ow * S.SW + D.kw * S.DW, D.c, D.n]) *
      cast(U, K[D.f, D.kd, D.kh, D.kw, D.c]))


@linalg_structured_op
def conv_3d_dhwcn_dhwcf(
    I=TensorDef(TV.T1, S.OD * S.SD + S.KD * S.DD, S.OH * S.SH + S.KH * S.DH,
                S.OW * S.SW + S.KW * S.DW, S.C, S.N),
    K=TensorDef(TV.T2, S.KD, S.KH, S.KW, S.C, S.F),
    O=TensorDef(U, S.OD, S.OH, S.OW, S.N, S.F, output=True),
    strides=AttributeDef(S.SD, S.SH, S.SW),
    dilations=AttributeDef(S.DD, S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.od, D.oh, D.ow, D.n, D.f, D.kd, D.kh, D.kw, D.c)
  O[D.od, D.oh, D.ow, D.n, D.f] += (
      cast(
          U, I[D.od * S.SD + D.kd * S.DD, D.oh * S.SH + D.kh * S.DH,
               D.ow * S.SW + D.kw * S.DW, D.c, D.n]) *
      cast(U, K[D.kd, D.kh, D.kw, D.c, D.f]))


@linalg_structured_op
def conv_3d_dhwcn_dhwfc(
    I=TensorDef(TV.T1, S.OD * S.SD + S.KD * S.DD, S.OH * S.SH + S.KH * S.DH,
                S.OW * S.SW + S.KW * S.DW, S.C, S.N),
    K=TensorDef(TV.T2, S.KD, S.KH, S.KW, S.F, S.C),
    O=TensorDef(U, S.OD, S.OH, S.OW, S.N, S.F, output=True),
    strides=AttributeDef(S.SD, S.SH, S.SW),
    dilations=AttributeDef(S.DD, S.DH, S.DW)):
  implements(ConvolutionOpInterface)
  domain(D.od, D.oh, D.ow, D.n, D.f, D.kd, D.kh, D.kw, D.c)
  O[D.od, D.oh, D.ow, D.n, D.f] += (
      cast(
          U, I[D.od * S.SD + D.kd * S.DD, D.oh * S.SH + D.kh * S.DH,
               D.ow * S.SW + D.kw * S.DW, D.c, D.n]) *
      cast(U, K[D.kd, D.kh, D.kw, D.f, D.c]))
