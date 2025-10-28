# =========================
# Build stage
# =========================
FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
WORKDIR /src

# Copy only the csproj first (faster restores)
COPY Services/BookRent.Identity/BookRent.Identity.csproj Services/BookRent.Identity/
RUN dotnet restore Services/BookRent.Identity/BookRent.Identity.csproj

# Copy the rest of the project
COPY Services/BookRent.Identity/ Services/BookRent.Identity/

# Publish
WORKDIR /src/Services/BookRent.Identity
RUN dotnet publish -c Release -o /app/publish

# =========================
# Runtime stage
# =========================
FROM mcr.microsoft.com/dotnet/aspnet:9.0
WORKDIR /app
COPY --from=build /app/publish .
ENV ASPNETCORE_URLS=http://+:8080
EXPOSE 8080
ENTRYPOINT ["dotnet", "BookRent.Identity.dll"]
