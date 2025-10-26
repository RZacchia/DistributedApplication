using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Design;

namespace BookRent.User.Infrastructure;

public sealed class UserDbContextFactory : IDesignTimeDbContextFactory<UserDbContext>
{
    public UserDbContext CreateDbContext(string[] args)
    {
        // Dev connection (will switch to Docker later)
        var cs = Environment.GetEnvironmentVariable("User_CS")
                 ?? "Server=localhost,1433;Database=UserDbDbDb;User Id=sa;Password=Your_password123;TrustServerCertificate=True;";

        var opts = new DbContextOptionsBuilder<UserDbContext>()
            .UseSqlServer(cs)
            .Options;

        return new UserDbContext(opts);
    }
}