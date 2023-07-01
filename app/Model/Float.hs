{-# LANGUAGE FlexibleInstances #-}

module Model.Float where

import Data.List

import Format

instance Format [Float] where
  format x = "[" ++ show (length x) ++ "]\n"
             ++ show (take 10 x)

instance Format [[Float]] where
  format x = "[" ++ show (length x) ++ " x " ++ show (length $ head x) ++ "]\n"
--             ++ unlines (map (("    " ++) . show . take 5) (take 5 x))
             ++ unlines (map showLine (take 5 x))
             ++ (if length x > 5 then "    ....(etc)" else "")
    where showLine v = (++ (if length v > 5 then " | ...." else "")) . ("    " ++) . intercalate " | " . map format . take 5 $ v

instance Format Float where
  format f =
    widen 13 ((if (f >= 0) then " " else "") ++ floatString)

    where floatString = show f
          widen i s = s ++ replicate (i-length s) ' '


