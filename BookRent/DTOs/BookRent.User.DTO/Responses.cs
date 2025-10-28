namespace BookRent.User.DTO;

public record UserDetailsResponse(Guid UserId, string UserName, string Email, string FirstName, string LastName);
public record UserFavouriteResponse(List<Guid> BooksIds);