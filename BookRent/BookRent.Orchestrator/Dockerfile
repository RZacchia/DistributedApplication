# =========================
# Build stage
# =========================
FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
WORKDIR /src

COPY BookRent.Orchestrator/BookRent.Orchestrator.csproj BookRent.Orchestrator/

RUN dotnet restore BookRent.Orchestrator/BookRent.Orchestrator.csproj

# Copy the rest of the project
COPY BookRent.Orchestrator/ BookRent.Orchestrator/

# Publish
WORKDIR /src/BookRent.Orchestrator
RUN dotnet publish -c Release -o /app/publish

# =========================
# Runtime stage
# =========================
FROM mcr.microsoft.com/dotnet/aspnet:9.0
WORKDIR /app
COPY --from=build /app/publish .
ENV ASPNETCORE_URLS=http://+:8080
EXPOSE 8080
ENTRYPOINT ["dotnet", "BookRent.Orchestrator.dll"]
