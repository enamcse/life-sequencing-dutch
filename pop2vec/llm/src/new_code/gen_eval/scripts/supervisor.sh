Oh. The letter was made. I am. Casting buddy. That's seminar total horizon high accident. I. I. That's it. That's at sequential learning is another. Hello, Nana Patekar. No, no Cortana. What's the suit? Sita? Network target targeted to put them. What is this when I work over the dish? I mean I can take a bit of a customer time. The current preparation has already. Should have comedy or probably like. They said. She took my second shot at the Jaguar. Phone details. Yeah, we did. British. Each other. I. Radio. I. Military force? I. Tell him. Pretty low. I'm near hotel. Well. That's right. I. Bharat Subhash. The rubber Enfield North Coast side of north. I don't know what remains. I. Coming. Do you have? Prime majority white American below 40 know. School. Teacher. I'm not. Australia yellow. OK. I. Hello. How about lunch? I. Maybe edibles? That's in the Governor OK. I. I can think what idea to generate his example. I want. To. Frederial approach at the Dostumi Gara. Hey God, Marie, hey. Hey. Cortana. Doing minute, they are the dominant Diana. Yeah, yeah. Who's this person now? Mutton, Abdul say. And now? Muslim. Husband. Actor 3. Duration Total 30 minutes. I. Jabra Cholte. Ratolte cholte. Minus stop. Minus 10. -. Tana. Yeah, I wanna dictate. So sad. Salon I go. Hello so long ago. Yeah, what's like? Any. Thing. To say. Call Maggie. OK OK. OK. Johno E Covetti Vette. OK. Down. Wake up. Wake up. What? Yeah. Yeah. Yeah. Hamkadesh mehsah. Larka GABA hoga. Joke Cortana, but a hoga. Let me see, let me see. 
#!/bin/bash
#SBATCH --job-name=supervisor
#SBATCH --partition=work_env
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=7-00:00:00
#SBATCH -o /projects/0/prjs1589/stonybrook/logs/supervisor-%j.out
#SBATCH -e /projects/0/prjs1589/stonybrook/logs/supervisor-%j.err

#
# Pipeline Supervisor SLURM Job
#
# This job runs continuously on the login node (work_env partition)
# and manages the entire generation/statistics pipeline.
#
# Features:
#   - Monitors SLURM queue for job status
#   - Submits generation jobs with GPU collision avoidance
#   - Automatically submits statistics jobs when generation completes
#   - Writes human-readable dashboard every 60 seconds
#   - Persists state to JSON for recovery
#
# Usage:
#   1. Edit supervisor_config.yaml with your models/experiments
#   2. Submit: sbatch supervisor.slurm
#   3. Monitor: cat supervisor_state/dashboard.txt
#   4. Stop: scancel <job_id>
#

echo "=========================================="
echo "Pipeline Supervisor"
echo "=========================================="
echo "Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo ""

# Change to script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Script directory: $SCRIPT_DIR"
echo "Config file: $SCRIPT_DIR/supervisor_config.yaml"
echo ""

# Load environment
echo "Loading environment..."
cd ~/life-sequencing-dutch/
source requirements/load_venv.sh
cd "$SCRIPT_DIR"

# Check config exists
if [[ ! -f "supervisor_config.yaml" ]]; then
    echo "ERROR: supervisor_config.yaml not found!"
    echo "Please create it first. See supervisor_config.yaml.example"
    exit 1
fi

echo ""
echo "Starting supervisor..."
echo "Dashboard will be written to: $SCRIPT_DIR/../supervisor_state/dashboard.txt"
echo ""
echo "=========================================="

# Run supervisor
python supervisor.py --config supervisor_config.yaml

echo ""
echo "=========================================="
echo "Supervisor finished: $(date)"
echo "=========================================="
