using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Design;

namespace BookRent.Identity.Infrastructure;

public sealed class IdentityDbContextFactory : IDesignTimeDbContextFactory<IdentityDbContext>
{
    public IdentityDbContext CreateDbContext(string[] args)
    {
          // Dev connection (will switch to Docker later)
          var cs = Environment.GetEnvironmentVariable("IDENTITY_CS")
                   ?? "Server=localhost,1433;Database=IdentityDb;User Id=sa;Password=Your_password123;TrustServerCertificate=True;";

          var opts = new DbContextOptionsBuilder<IdentityDbContext>()
            .UseSqlServer(cs)
            .Options;

          return new IdentityDbContext(opts);
    }   
}