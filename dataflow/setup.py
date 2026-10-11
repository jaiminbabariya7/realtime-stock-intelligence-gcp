"""Packages the pipeline's helper modules so Dataflow workers can import them.

Used automatically by stock_pipeline.py (--setup_file).
"""
import setuptools

setuptools.setup(
    name="stock-dataflow-pipeline",
    version="1.1.0",
    py_modules=["features"],
)
