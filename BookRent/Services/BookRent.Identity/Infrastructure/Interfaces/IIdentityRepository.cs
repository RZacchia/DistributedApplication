using BookRent.Identity.Models;

namespace BookRent.Identity.Infrastructure.Interfaces;

public interface IIdentityRepository
{
    Task<Role> GetUserRoleAsync(Guid userId);
    Task<bool> IsUserInRoleOrHigherAsync(UserRole userRole);
    Task<Guid?> GetUserIdAsync(string userName, string password);
    Task<bool> RegisterUserAsync(UserCredentials userCredentials);
    Task<bool> AddUserRoleAsync(UserRole userRole);

}