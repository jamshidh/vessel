{-# LANGUAGE OverloadedStrings #-}

module Model.Token (
  Token,
  tokenize,
  tokensToTokenTrie
  ) where

import Data.Binary
import Data.Binary.Get
import qualified Data.ByteString.Char8 as BC
import Data.Trie (Trie)
import qualified Data.Trie as Trie

import Format

newtype Token = Token String deriving (Show)

instance Format Token where
  format (Token t) = t



instance Binary Token where
  get = do
    size <- getInt32le
    tokenString <- getByteString $ fromIntegral size
    return $ Token $ BC.unpack tokenString
  put = undefined


tokenize :: Trie Int -> BC.ByteString -> [Int]
tokenize _ "" = []
tokenize theTrie s =
  case Trie.match theTrie s of
    Nothing -> error $ "can't find token to match to: " ++ show s
    Just (_, i, rest) -> i:tokenize theTrie rest

tokensToTokenTrie :: [Token] -> Trie Int
tokensToTokenTrie tokens = Trie.fromList $ reverse $ zip (map tokenAsByteString tokens) [0..]
  where tokenAsByteString (Token s) = BC.pack s


