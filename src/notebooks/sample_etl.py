# Databricks notebook source
# A minimal package smoke test for the platform SDK bundle.
from aai_core import __version__

print(f"aai-core {__version__}: package import verified")
