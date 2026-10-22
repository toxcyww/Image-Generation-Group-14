"""V2 CLI: joint optimization plus the non-ID condition baseline."""
import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description='V2: frozen StyleGAN2 joint optimization and non-ID baseline (300 steps each)')
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--input', type=Path, help='Square cropped face image or a directory of such images')
    inputs.add_argument('--latent', type=Path, help='Render a saved [1,18,512] latent; no re-optimization')
    parser.add_argument('--output', type=Path, required=True, help='A NEW directory for input mode; a NEW PNG path for latent mode')
    parser.add_argument('--teacher', type=Path, help='Optional saved teacher for a single image; default: automatic selection')
    parser.add_argument('--seed', type=int, help='Optional full donor seed; default: 20261020 + numeric filename, or content hash')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists. Choose a new output path; existing results are preserved.')
    if args.latent and (args.teacher or args.seed is not None):
        parser.error('--teacher/--seed apply only to image inputs')
    if args.input and not args.input.exists():
        parser.error('Input path does not exist')
    if args.latent and not args.latent.is_file():
        parser.error('Latent file does not exist')
    if args.teacher and not args.teacher.is_file():
        parser.error('Teacher file does not exist')
    if args.input and args.input.is_dir() and args.teacher:
        parser.error('--teacher supports one input image at a time')
    images = None
    if args.input and args.input.is_dir():
        images = sorted(p for p in args.input.iterdir() if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.webp'})
        if not images:
            parser.error('Input directory has no supported images')
        names = [p.stem.casefold() for p in images]
        if len(names) != len(set(names)):
            parser.error('Batch input filenames must have distinct stems')
    from anonymizer import Anonymizer, write_json
    model = Anonymizer(args.device)
    if args.latent:
        model.render_latent(args.latent, args.output)
        print(f'Saved: {args.output.resolve()}')
    elif images is None:
        model.anonymize(args.input, args.output, seed=args.seed, teacher_path=args.teacher)
        print(f'Saved: {(args.output / "final.png").resolve()}')
    else:
        args.output.mkdir(parents=True)
        records = []
        for image in images:
            try:
                record = model.anonymize(image, args.output / image.stem, seed=args.seed)
            except Exception as error:
                record = {'status': 'failed', 'error': type(error).__name__, 'message': str(error)}
            records.append({'input': image.name, **record})
            write_json(args.output / 'batch_results.json', records)
        if any(r['status'] != 'completed' for r in records):
            print('Batch finished with failures; see batch_results.json', file=sys.stderr)
            return 1
        print(f'Saved {len(records)} images: {args.output.resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
