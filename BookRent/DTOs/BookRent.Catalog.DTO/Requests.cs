namespace BookRent.Catalog.DTO;

public record BookSearchRequest(string Name);
public record AddBookRequest(string Name, string Description, string Author, string Isbn);
public record UpdateBookRequest(Guid Id, string Name, string Description, string Author, string Isbn, bool IsVisible);