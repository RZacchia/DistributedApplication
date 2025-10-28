using BookRent.Orchestrator.Clients;
using Scalar.AspNetCore;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
// Learn more about configuring OpenAPI at https://aka.ms/aspnet/openapi
builder.Services.AddOpenApi();
var cfg = builder.Configuration;

builder.Services.AddHttpClient<CatalogClient>(client =>
{
    string uri = Environment.GetEnvironmentVariable("CATALOG_URL")
                 ?? cfg["Services:CatalogServiceUrl"]!;
    client.BaseAddress = new Uri(uri);
    
});

builder.Services.AddHttpClient<IdentityClient>(client =>
{
    string uri = Environment.GetEnvironmentVariable("IDENTITY_URL")
                 ?? cfg["Services:IdentityServiceUrl"]!;
    client.BaseAddress = new Uri(uri);
    
});
builder.Services.AddHttpClient<RentingClient>(client =>
{
    string uri = Environment.GetEnvironmentVariable("RENTING_URL")
                 ?? cfg["Services:RentingServiceUrl"]!;
    client.BaseAddress = new Uri(uri);
});
builder.Services.AddHttpClient<UserClient>(client =>
{
    string uri = Environment.GetEnvironmentVariable("USER_URL")
                 ?? cfg["Services:UserServiceUrl"]!;
    client.BaseAddress = new Uri(uri);
});

var app = builder.Build();


// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.MapScalarApiReference();
    app.MapOpenApi();
}

app.Run();

