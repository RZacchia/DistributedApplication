using BookRent.Identity.Api;
using BookRent.Identity.Infrastructure;
using BookRent.Identity.Infrastructure.Interfaces;

var builder = WebApplication.CreateBuilder(args);
builder.Configuration.AddEnvironmentVariables();

// Add services to the container.
// Learn more about configuring OpenAPI at https://aka.ms/aspnet/openapi
builder.Services.AddOpenApi();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddDbContext<IdentityDbContext>();
builder.Services.AddScoped<IIdentityRepository, IdentityRepository>();


var app = builder.Build();
app.MapIdentityEndPoints();

if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
}

app.UseHttpsRedirection();



app.Run();

