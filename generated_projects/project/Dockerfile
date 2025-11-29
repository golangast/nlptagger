FROM golang:1.22-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN go build -o webserver server.go

FROM alpine:latest
WORKDIR /app
COPY --from=builder /app/webserver .
COPY --from=builder /app/data/orchestrator.db ./data/orchestrator.db
EXPOSE 8080
CMD ["./webserver"]
