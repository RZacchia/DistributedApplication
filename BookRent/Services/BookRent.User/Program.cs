using BookRent.User.Api;
using BookRent.User.Infrastructure;
using BookRent.User.Infrastructure.Interfaces;
using Microsoft.EntityFrameworkCore;
using Scalar.AspNetCore;

var builder = WebApplication.CreateBuilder(args);
var cs = Environment.GetEnvironmentVariable("ConnectionStrings__UserDb")
         ?? builder.Configuration.GetConnectionString("UserDb");

// Add services to the container.
// Learn more about configuring OpenAPI at https://aka.ms/aspnet/openapi
builder.Services.AddOpenApi();
builder.Services.AddDbContext<UserDbContext>(opt =>
    opt.UseSqlServer(cs, sql => sql.EnableRetryOnFailure(5)));
builder.Services.AddScoped<IUserRepository, UserRepository>();


var app = builder.Build();

using (var scope = app.Services.CreateScope())
{
    var db = scope.ServiceProvider.GetRequiredService<UserDbContext>();
    db.Database.Migrate();
}



if (app.Environment.IsDevelopment())
{
    app.MapScalarApiReference();
    app.MapOpenApi();
}
app.MapUserEndpoints();
app.UseHttpsRedirection();



app.Run();