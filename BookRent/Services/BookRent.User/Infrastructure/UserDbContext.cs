using BookRent.User.Models;
using Microsoft.EntityFrameworkCore;

namespace BookRent.User.Infrastructure;

public class UserDbContext(DbContextOptions<UserDbContext> options) : DbContext(options)
{
    public DbSet<UserBaseData> UserData => Set<UserBaseData>();
    public DbSet<UserFavourites> UserFavourites => Set<UserFavourites>();

    protected override void OnModelCreating(ModelBuilder b)
    {
        
    }
    
}