# =========================
# Build stage
# =========================
FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
WORKDIR /src

# Copy only the csproj first (faster restores)
COPY BookRent.Gateway/BookRent.Gateway.csproj BookRent.Gateway/
RUN dotnet restore BookRent.Gateway/BookRent.Gateway.csproj

# Copy the rest of the project
COPY BookRent.Gateway/ BookRent.Gateway/

# Publish
WORKDIR /src/BookRent.Gateway
RUN dotnet publish -c Release -o /app/publish

# =========================
# Runtime stage
# =========================
FROM mcr.microsoft.com/dotnet/aspnet:9.0
WORKDIR /app
COPY --from=build /app/publish .
ENV ASPNETCORE_URLS=http://+:8080
EXPOSE 8080
ENTRYPOINT ["dotnet", "BookRent.Gateway.dll"]
