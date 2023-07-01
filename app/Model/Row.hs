
module Model.Row where

import Data.ByteString (ByteString)

data Row =
  QuantizedRow ByteString |
  Row [Float]

