using BookRent.Identity.DTO.Enums;
using BookRent.Identity.Models;

namespace BookRent.Identity.Infrastructure.Interfaces;

internal interface IIdentityRepository
{
    Task<Role?> GetUserRoleAsync(Guid userId);
    Task<bool> IsUserInRoleOrHigherAsync(UserRole userRole);
    Task<Guid?> GetUserIdAsync(string userName, string password);
    Task<bool> RegisterUserAsync(UserCredentials userCredentials,  UserRole userRole);

}