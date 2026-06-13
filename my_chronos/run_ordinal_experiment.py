import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-ordinal-head", action="store_true")
    parser.add_argument("--distance-weight", type=float, default=1.0)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--max-series", type=int, default=50)
    args = parser.parse_args()
    
    cmd = ["python", "train_ordinal.py", "--config", "ordinal_config.yaml"]
    
    if args.use_ordinal_head:
        cmd.append("--use-ordinal-head")
        cmd.extend(["--distance-weight", str(args.distance_weight)])
    if args.num_epochs:
        cmd.extend(["--num-epochs", str(args.num_epochs)])
    if args.batch_size:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if args.learning_rate:
        cmd.extend(["--learning-rate", str(args.learning_rate)])
    
    env = {"OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "TORCH_NUM_THREADS": "8"}
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, env={**subprocess.os.environ, **env}, check=True)
    except KeyboardInterrupt:
        print("\n⚠️ Experiment interrupted")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Experiment failed with error code {e.returncode}")

if __name__ == "__main__":
    main()