# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## 0.0.2 - 2017-11-12
### Added
- Implemented reconstruction loss.
- Saving reconstructed image as file.
- Bug fixes:
    - Missing square in equation 4 for margin (class) loss.
    - More intuitive variable naming.
- Resolve Pylint warnings and reformat code.
- Improve training speed by using PyTorch DataParallel to wrap our model.
    - PyTorch will parallelized the model and data over multiple GPUs.
- Supports training:
    - on CPU (tested with macOS Sierra)
    - on one GPU (tested with 1 Tesla K80 GPU)
    - on multiple GPU (tested with 8 GPUs)
    - with or without CUDA (tested with CUDA version 8.0.61)
    - cuDNN 5 (tested with cuDNN 5.1.3)

## 0.0.1 - 2017-11-04
### Added
- Initial release. The first beta version. API is stable. The code runs. So, I think it's safe to use for development but not ready for general production usage.

[Unreleased]: https://github.com/olivierlacan/keep-a-changelog/compare/v1.0.0...HEAD
[0.0.2]: https://github.com/cedrickchee/keep-a-changelog/compare/v0.0.1...v0.0.2
