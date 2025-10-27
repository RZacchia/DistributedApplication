using BookRent.User.Api;
using BookRent.User.Infrastructure;
using BookRent.User.Infrastructure.Interfaces;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
// Learn more about configuring OpenAPI at https://aka.ms/aspnet/openapi
builder.Services.AddOpenApi();
builder.Services.AddDbContext<UserDbContext>();
builder.Services.AddScoped<IUserRepository, UserRepository>();


var app = builder.Build();
app.MapUserEndpoints();
if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
}

app.UseHttpsRedirection();



app.Run();