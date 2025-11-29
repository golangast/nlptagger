# Simple Go Webserver

This project contains a minimal Go HTTP server and a Dockerfile to containerize it.

## Build & Run

```sh
go run server.go
```

Or with Docker:

```sh
docker build -t go-webserver .
 docker run -p 8080:8080 go-webserver
```
