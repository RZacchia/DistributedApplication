# =========================
# Build stage
# =========================
FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
WORKDIR /src

# Copy only the csproj first (faster restores)
COPY Services/BookRent.Catalog/BookRent.Catalog.csproj Services/BookRent.Catalog/
RUN dotnet restore Services/BookRent.Catalog/BookRent.Catalog.csproj

# Copy the rest of the project
COPY Services/BookRent.Catalog/ Services/BookRent.Catalog/

# Publish
WORKDIR /src/Services/BookRent.Catalog
RUN dotnet publish -c Release -o /app/publish

# =========================
# Runtime stage
# =========================
FROM mcr.microsoft.com/dotnet/aspnet:9.0
WORKDIR /app
COPY --from=build /app/publish .
ENV ASPNETCORE_URLS=http://+:8081
EXPOSE 8081
ENTRYPOINT ["dotnet", "BookRent.Catalog.dll"]
