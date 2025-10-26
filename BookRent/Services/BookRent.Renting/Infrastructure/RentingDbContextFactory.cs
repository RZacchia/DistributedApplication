using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Design;

namespace BookRent.Renting.Infrastructure;

public sealed class RentingDbContextFactory : IDesignTimeDbContextFactory<RentingDbContext>
{
    public RentingDbContext CreateDbContext(string[] args)
    {
        // Dev connection (will switch to Docker later)
        var cs = Environment.GetEnvironmentVariable("RENTING_CS")
                 ?? "Server=localhost,1433;Database=RentingDbDb;User Id=sa;Password=Your_password123;TrustServerCertificate=True;";

        var opts = new DbContextOptionsBuilder<RentingDbContext>()
            .UseSqlServer(cs)
            .Options;

        return new RentingDbContext(opts);
    }
}