using BookRent.Identity.Infrastructure.Interfaces;
using BookRent.Identity.Models;
using Microsoft.EntityFrameworkCore;

namespace BookRent.Identity.Infrastructure;

public class IdentityRepository : IIdentityRepository
{
    private IdentityDbContext Context { get; init; }
    public IdentityRepository(IdentityDbContext context)
    {
        Context = context;
    }
    
    
    public async Task<Role> GetUserRoleAsync(Guid userId)
    {
        return await Context.UserRoles
            .Where(u => u.UserId == userId)
            .Select(u => u.Role)
            .FirstAsync();
    }

    public async Task<bool> IsUserInRoleOrHigherAsync(UserRole userRole)
    {
        return await Context.UserRoles.AnyAsync(u => u.UserId == userRole.UserId && u.Role >= userRole.Role);
    }

    

    public async Task<Guid?> GetUserIdAsync(string userName, string password)
    {
        return await Context.UserCredentials
            .Where(u => u.Email == userName && u.Password == password)
            .Select(u => u.UserId)
            .FirstOrDefaultAsync();
    }

    public async Task<bool> RegisterUserAsync(UserCredentials userCredentials)
    {
        await Context.UserCredentials.AddAsync(userCredentials);
        return await Context.SaveChangesAsync() > 0;
    }

    public async Task<bool> AddUserRoleAsync(UserRole userRole)
    {
        await Context.UserRoles.AddAsync(userRole);
        return await Context.SaveChangesAsync() > 0;
    }
}