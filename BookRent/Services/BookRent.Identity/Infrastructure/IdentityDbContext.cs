using BookRent.Identity.Models;
using Microsoft.EntityFrameworkCore;

namespace BookRent.Identity.Infrastructure;

public sealed class IdentityDbContext(DbContextOptions<IdentityDbContext> options) : DbContext(options)
{
    public DbSet<UserRole> UserRoles => Set<UserRole>();

    protected override void OnModelCreating(ModelBuilder b)
    {
        
    }
}