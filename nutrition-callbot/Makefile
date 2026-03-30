.PHONY: up down build logs dev restart clean ps health

# === Production ===
up:
	docker compose up -d

down:
	docker compose down

build:
	docker compose build

logs:
	docker compose logs -f

restart:
	docker compose restart

ps:
	docker compose ps

# === Development (with hot-reload) ===
dev:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

dev-logs:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml logs -f

dev-down:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml down

# === Utilities ===
clean:
	docker compose down -v --rmi local
	docker system prune -f

health:
	@echo "=== Gateway Health ==="
	@curl -s http://localhost:8000/health | python3 -m json.tool || echo "Gateway is DOWN"
	@echo "\n=== ASR Health ==="
	@curl -s http://localhost:50051/health | python3 -m json.tool || echo "ASR is DOWN"
	@echo "\n=== Brain Health ==="
	@curl -s http://localhost:50052/health | python3 -m json.tool || echo "Brain is DOWN"
	@echo "\n=== TTS Health ==="
	@curl -s http://localhost:50053/health | python3 -m json.tool || echo "TTS is DOWN"
