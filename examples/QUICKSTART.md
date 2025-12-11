# SecureFL Docker Quick Start

Get up and running with SecureFL federated learning in under 5 minutes!

## 🚀 One-Command Start

```bash
# Start federated learning with 1 server + 3 clients
make start
```

Or using the script:
```bash
./run_training.sh start
```

Or using Docker Compose directly:
```bash
docker-compose up
```

## ⚡ Quick Commands

| Command | Description |
|---------|-------------|
| `make start` | Start training (foreground) |
| `make start-bg` | Start training (background) |
| `make logs` | View all logs |
| `make status` | Check container status |
| `make stop` | Stop training |
| `make clean` | Clean up everything |

## 📋 Prerequisites

- Docker & Docker Compose installed
- 4GB+ RAM available
- Internet connection (to pull image)

## 🔍 What Happens

1. **Server starts** on port 8080
2. **3 clients connect** with different MNIST data partitions
3. **10 training rounds** with ZKP verification
4. **Results saved** to `./results/` directory

## 📊 Monitoring Progress

**View logs in real-time:**
```bash
make logs-follow
```

**Check specific services:**
```bash
make logs-server    # Server only
make logs-client1   # Client 1 only
make status         # All containers
```

## ⚙️ Quick Customization

**Change training rounds:**
Edit `docker-compose.yml`, line 21:
```yaml
"--rounds", "5"    # Change from "10" to "5"
```

**Use CIFAR-10 instead of MNIST:**
Edit `docker-compose.yml`, line 25:
```yaml
"--model", "cifar10"   # Change from "mnist"
```

And update all clients (lines ~53, ~82, ~111):
```yaml
"--dataset", "cifar10"  # Change from "mnist"
```

**Disable ZKP for faster training:**
Remove `"--enable-zkp"` from server and all clients

## 🎯 Expected Output

```
✅ Server: Waiting for clients...
✅ Client 1: Connected, partition 0
✅ Client 2: Connected, partition 1  
✅ Client 3: Connected, partition 2
🔄 Round 1/10: Training...
🔐 ZKP verification: ✓ Passed
📈 Accuracy: 85.2%
...
🎉 Training complete!
```

## 🔧 Troubleshooting

**Server won't start?**
```bash
make logs-server
```

**Clients can't connect?**
```bash
# Check if server is healthy
docker ps
# Look for "healthy" status
```

**Out of memory?**
```bash
# Reduce batch size in docker-compose.yml
"--batch-size", "16"  # Change from "32"
```

**ZKP errors?**
```bash
# Disable ZKP temporarily
# Remove "--enable-zkp" from commands
```

## 📁 Generated Files

After running, you'll see:
```
examples/
├── data/          # MNIST dataset + partitions
├── logs/          # Training logs
├── results/       # Model checkpoints & metrics
└── ...
```

## 🎮 Next Steps

1. **Try different models**: Change `--model` to `cifar10`
2. **Scale up**: Add more clients to `docker-compose.yml`
3. **Custom config**: Use `config.yaml` for advanced settings
4. **Monitor resources**: Run `make stats`
5. **Experiment**: Modify hyperparameters and compare results

## 💡 Pro Tips

- Run `make start-bg` to train in background while working
- Use `make logs-follow` to watch training progress
- Check `make help` for all available commands
- Results are saved and persist between runs

---

**Need help?** Check the full [README.md](README.md) for detailed documentation.

**Ready to customize?** See [config.yaml](config.yaml) for all options.