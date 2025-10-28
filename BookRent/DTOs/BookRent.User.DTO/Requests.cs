namespace BookRent.User.DTO;

public record UserDetailsRequest(Guid UserId, string UserName, string FirstName, string LastName, string Email);