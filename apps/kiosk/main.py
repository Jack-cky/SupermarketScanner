import argparse

from smktscnr import SupermarketScanner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run supermarket scanner inference from the camera",
    )
    parser.add_argument(
        "--weights",
        default="checkpoints/smktscnr_ft.onnx",
        help="Path to model weights (default: checkpoints/smktscnr_ft.onnx)",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    scanner = SupermarketScanner(weights=args.weights)
    return 0 if scanner.checkout() else 1


if __name__ == "__main__":
    raise SystemExit(main())
