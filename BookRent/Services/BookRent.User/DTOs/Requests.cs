namespace BookRent.User.DTOs;

public record UserDetailsRequest(Guid UserId, string UserName, string FirstName, string LastName, string Email);